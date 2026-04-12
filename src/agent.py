"""
LangGraph Agent Definition for Email RAG System.
Includes nodes for retrieving context, drafting, evaluating, and rewriting.
"""
import json
from typing import TypedDict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import chromadb
from chromadb.utils import embedding_functions

from src.llm_env import load_llm_env
from src.ingest_from_text import _ROOT

class AgentState(TypedDict):
    user_inputs: dict
    email_history: str
    current_draft: str
    feedback: str
    evaluate_status: str
    iterations: int
    chat_history: list  # List of {"role": "user"|"assistant", "content": str}

env = load_llm_env()

llm_chat = ChatOpenAI(
    model=env.chat_model,
    api_key=env.api_key,
    base_url=env.base_url,
    temperature=0.7 
)

llm_critic = ChatOpenAI(
    model=env.chat_model,
    api_key=env.api_key,
    base_url=env.base_url,
    temperature=0.0 
)

def retrieve_node(state: AgentState):
    contact_id = state["user_inputs"].get("contact_id", -1)
    
    chroma_path = _ROOT / "chroma_data"
    emb_kwargs = {"api_key": env.api_key, "model_name": env.embedding_model}
    if env.base_url:
        emb_kwargs["api_base"] = env.base_url
        
    ef = embedding_functions.OpenAIEmbeddingFunction(**emb_kwargs)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name="email_chunks", embedding_function=ef)
    
    goal = state["user_inputs"].get("goal", "")
    
    history_text = ""
    # Chỉ load từ db nếu contact_id hợp lệ
    if contact_id != -1:
        try:
            results = collection.query(
                query_texts=[goal],
                n_results=3,
                where={"contact_id": int(contact_id)}
            )
            if results and results.get("documents") and results["documents"][0]:
                history_text = "\n\n---\n\n".join(results["documents"][0])
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            history_text = ""
            
    return {"email_history": history_text, "iterations": state.get("iterations", 0)}

def draft_node(state: AgentState):
    system_prompt = "Bạn là trợ lý viết email chuyên nghiệp, súc tích và hiệu quả. Bạn nhớ toàn bộ ngữ cảnh cuộc hội thoại hiện tại."
    
    ui = state["user_inputs"]
    chat_history = state.get("chat_history", [])

    # Build conversation history block for prompt context
    history_block = ""
    if chat_history:
        history_block = "--- LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ ---\n"
        for turn in chat_history:
            role_label = "Người dùng" if turn["role"] == "user" else "AI"
            history_block += f"[{role_label}]: {turn['content']}\n\n"
        history_block += "--- KẾT THÚC LỊCH SỬ ---\n"

    prompt = f"""
{history_block}
Yêu cầu hiện tại của người dùng:
- Tên khách hàng: {ui.get('name', 'Unknown')}
- Công ty: {ui.get('company', 'Unknown')}
- Ngành nghề: {ui.get('industry', 'Unknown')}
- Mục tiêu/Yêu cầu: {ui.get('goal', '')}
- Mục tiêu email: {ui.get('objective', '') or 'Tùy theo nội dung yêu cầu'}
- Giọng điệu: {ui.get('tone', '') or 'Tự nhiên, phù hợp ngữ cảnh'}

Lịch sử giao dịch liên quan từ cơ sở dữ liệu (tham khảo thêm nếu có):
{state['email_history'] or '(Không có lịch sử giao dịch)'}

Hãy viết hoặc chỉnh sửa email dựa trên tất cả thông tin trên. Nếu người dùng đang yêu cầu thay đổi phiên bản cũ (ví dụ: viết ngắn lại, thêm CTA...), hãy dựa vào phiên bản AI đã viết trong lịch sử hội thoại và chỉnh sửa theo. CHỈ trả về nội dung email, không kèm giải thích.
"""
    
    response = llm_chat.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    return {"current_draft": response.content}

def evaluate_node(state: AgentState):
    draft = state["current_draft"]
    
    system_prompt = "Bạn là chuyên gia thẩm định chất lượng email. Nhiệm vụ của bạn là kiểm tra xem email có thỏa mãn các yêu cầu cơ bản hay không."
    
    prompt = f"""
Hãy đánh giá bản nháp email sau đây một cách tổng quan và linh hoạt. 
Tiêu chí kiểm duyệt (LINH HOẠT):
1. Có nên chứa Call-to-Action (CTA) không? Nếu mục tiêu không đòi hỏi, có thể bỏ qua. Nếu cần, hãy yêu cầu thêm.
2. Từ ngữ chuẩn mực, không quá dài dòng.
3. Không quá nhiều từ ngữ sáo rỗng.

Bản nháp email:
{draft}

Hãy trả về kết quả theo định dạng JSON với 2 khóa:
{{
  "status": "OK" hoặc "REJECTED",
  "feedback": "Lý do ngắn gọn nếu bị REJECTED. Nếu OK thì cứ để trống."
}}

Ghi chú: Nếu email đã khá ổn hoặc cơ bản xài được, hãy trả về "status": "OK" để giữ cho ứng dụng nhanh nhạy, không bắt người dùng đợi lâu.
"""

    response = llm_critic.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    try:
        content = response.content
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        parsed = json.loads(content.strip())
        status = parsed.get("status", "OK")
        feedback = parsed.get("feedback", "")
    except Exception as e:
        print(f"Evaluate parse error: {e}")
        status = "OK"  # Fallback to avoid infinite loop
        feedback = ""
        
    return {"evaluate_status": status, "feedback": feedback, "iterations": state["iterations"] + 1}

def rewrite_node(state: AgentState):
    draft = state["current_draft"]
    feedback = state["feedback"]
    
    system_prompt = "Bạn là trợ lý viết và chỉnh sửa email xuất sắc."
    
    prompt = f"""
Đây là bản nháp email hiện tại của tôi:
{draft}

Nhận xét/Góp ý từ người kiểm duyệt:
{feedback}

Dựa vào nhận xét trên, hãy SỬA LẠI email này cho tốt hơn. Viết lại toàn bộ bức thư sao cho mượt mà, chuyên nghiệp và CHỈ trả về nội dung email cuối cùng (không kèm giải thích).
"""
    
    response = llm_chat.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    return {"current_draft": response.content}

def router_node(state: AgentState):
    # Safety mechanism to prevent infinite loops (max 3 evaluations)
    if state["evaluate_status"] == "OK" or state["iterations"] >= 3:
        return "end"
    else:
        return "rewrite"

# Define the graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("draft", draft_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("rewrite", rewrite_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "draft")
workflow.add_edge("draft", "evaluate")
workflow.add_conditional_edges(
    "evaluate", 
    router_node,
    {
        "end": END,
        "rewrite": "rewrite"
    }
)
workflow.add_edge("rewrite", "evaluate")

# Compile explicitly
email_agent_app = workflow.compile()
