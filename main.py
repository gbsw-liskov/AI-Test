import json
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from fastapi import APIRouter, FastAPI, Form, UploadFile
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API key via environment variable.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def format_property_info(
    property_id: int,
    name: str,
    address: str,
    property_type: str,
    floor: int,
    build_year: int,
    area: int,
    available_date: str,
    market_price: Optional[float] = None,
    deposit: Optional[float] = None,
    monthly_rent: Optional[float] = None,
) -> str:
    """Create a compact property summary for prompts."""
    lines = [
        f"매물 ID: {property_id}",
        f"이름: {name}",
        f"주소: {address}",
        f"유형: {property_type}",
        f"층수: {floor}",
        f"준공연도: {build_year}",
        f"면적: {area}",
        f"입주 가능일: {available_date}",
    ]
    if market_price is not None:
        lines.append(f"시세: {market_price}")
    if deposit is not None:
        lines.append(f"보증금: {deposit}")
    if monthly_rent is not None:
        lines.append(f"월세: {monthly_rent}")
    return "\n".join(lines)


async def read_files(files: Optional[List[UploadFile]]) -> str:
    """Read uploaded files as text and label them; tolerate encoding issues."""
    if not files:
        return "첨부 문서 없음"

    chunks: List[str] = []
    for file in files:
        try:
            content = (await file.read()).decode("utf-8", errors="ignore")
            chunks.append(f"[{file.filename}] 내용:\n{content}")
        except Exception:
            chunks.append(f"[{file.filename}] 파일을 읽을 수 없습니다.")
    return "\n\n".join(chunks)


def call_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("환경변수 GEMINI_API_KEY가 설정되지 않았습니다.")
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        user_prompt,
        generation_config={"temperature": temperature},
    )
    return response.text or ""


def parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


app = FastAPI(title="Gemini Test Server")
router = APIRouter()


class ChecklistRequest(BaseModel):
    propertyId: int
    name: str
    address: str
    propertyType: str
    floor: int
    buildYear: int
    area: int
    availableDate: str


@router.post("/checklist")
async def generate_checklist(data: ChecklistRequest):
    property_block = format_property_info(
        property_id=data.propertyId,
        name=data.name,
        address=data.address,
        property_type=data.propertyType,
        floor=data.floor,
        build_year=data.buildYear,
        area=data.area,
        available_date=data.availableDate,
    )

    system_prompt = (
        "너는 부동산 위험 사전점검 전문가다. "
        "입지/건물 상태/법적 리스크/가격 및 수익성/계약/관리 관점에서 "
        "구체적인 확인 질문을 만든다. "
        "응답은 JSON만 반환하고 다른 문장, 마크다운, 코드블록, 백틱을 절대 포함하지 마라. "
        "스키마: {\"contents\": [\"질문 또는 체크포인트\", ...]} "
        "불필요한 설명·번호 매기기·여는/닫는 텍스트 금지."
    )

    user_prompt = (
        "다음 매물 정보를 기반으로 반드시 필요한 체크리스트를 8~12개 작성해라.\n\n"
        f"{property_block}"
    )

    try:
        output = call_gemini(system_prompt, user_prompt, temperature=0.25)
    except Exception as exc:
        return {"error": f"Gemini 호출 실패: {exc}"}

    parsed = parse_json(output)
    if isinstance(parsed, dict) and isinstance(parsed.get("contents"), list):
        return {"contents": parsed["contents"]}

    fallback_items = [line.strip("- ").strip() for line in output.splitlines() if line.strip()]
    return {"contents": fallback_items}


@router.post("/analyze")
async def analyze_property(
    propertyId: int = Form(...),
    name: str = Form(...),
    address: str = Form(...),
    propertyType: str = Form(...),
    floor: int = Form(...),
    buildYear: int = Form(...),
    area: int = Form(...),
    availableDate: str = Form(...),
    marketPrice: float = Form(...),
    deposit: float = Form(...),
    monthlyRent: float = Form(...),
    files: Optional[List[UploadFile]] = None,
):
    property_block = format_property_info(
        property_id=propertyId,
        name=name,
        address=address,
        property_type=propertyType,
        floor=floor,
        build_year=buildYear,
        area=area,
        available_date=availableDate,
        market_price=marketPrice,
        deposit=deposit,
        monthly_rent=monthlyRent,
    )
    attachments = await read_files(files)

    system_prompt = (
        "너는 부동산 리스크 분석 전문가다. "
        "입지, 건물 물리적 상태, 법적 리스크(권리, 인허가, 임대차), "
        "가격 및 수익성, 계약/운영 리스크를 종합 평가한다. "
        "응답은 아래 JSON 스키마만 사용하고, 마크다운·코드블록·백틱·주석 등 JSON 외 텍스트를 절대 포함하지 마라.\n"
        "{"
        "\"totalRisk\": 0~100 사이 정수,"
        "\"summary\": \"핵심 위험 요약(2문장 이내)\","
        "\"details\": ["
        "{"
        "\"title\": \"위험 항목 제목\","
        "\"content\": \"근거와 영향, 확인/완화 필요 조치\","
        "\"severity\": \"low|medium|high\""
        "}"
        "]"
        "}"
        "severity는 영향도와 시급성을 반영하여 high/medium/low 중 하나로만 표기한다. "
        "문장 앞에 불릿/번호를 붙이지 말고 JSON 외 텍스트를 추가하지 마라."
    )

    user_prompt = (
        "다음 매물 정보를 검토하고 위험도를 산출해라. "
        "첨부 문서 내용도 근거로 활용하라.\n\n"
        f"{property_block}\n\n"
        f"첨부 문서:\n{attachments}"
    )

    try:
        output = call_gemini(system_prompt, user_prompt, temperature=0.2)
    except Exception as exc:
        return {"error": f"Gemini 호출 실패: {exc}"}

    parsed = parse_json(output)
    if parsed:
        return parsed
    return {"raw_output": output}


@router.post("/solution")
async def propose_solution(
    propertyId: int = Form(...),
    name: str = Form(...),
    address: str = Form(...),
    propertyType: str = Form(...),
    floor: int = Form(...),
    buildYear: int = Form(...),
    area: int = Form(...),
    availableDate: str = Form(...),
    marketPrice: float = Form(...),
    deposit: float = Form(...),
    monthlyRent: float = Form(...),
    totalRisk: float = Form(...),
    summary: str = Form(...),
    details: str = Form(...),
    files: Optional[List[UploadFile]] = None,
):
    property_block = format_property_info(
        property_id=propertyId,
        name=name,
        address=address,
        property_type=propertyType,
        floor=floor,
        build_year=buildYear,
        area=area,
        available_date=availableDate,
        market_price=marketPrice,
        deposit=deposit,
        monthly_rent=monthlyRent,
    )
    attachments = await read_files(files)
    parsed_details: Any = parse_json(details) or details

    system_prompt = (
        "너는 부동산 컨설팅 전문가다. "
        "주어진 위험 요약을 바탕으로 실행 가능한 대처 방안과 확인 체크리스트를 제안한다. "
        "응답은 JSON만 반환하고 다른 텍스트, 마크다운, 코드블록, 백틱을 절대 포함하지 마라.\n"
        "{"
        "\"coping\": ["
        "{"
        "\"title\": \"대처 전략 제목\","
        "\"actions\": [\"구체적 실행 단계\"]"
        "}"
        "],"
        "\"checklist\": [\"후속 확인 항목\"]"
        "}"
        "actions는 2~5개의 짧은 단계로 작성하며, 바로 실행할 수 있게 작성한다."
    )

    user_prompt = (
        "다음 매물의 위험 요약과 세부 내용을 바탕으로 맞춤형 대처 방안을 제안해라. "
        "첨부 문서에서 근거가 보이면 반영하라.\n\n"
        f"{property_block}\n\n"
        f"총 위험도: {totalRisk}\n"
        f"요약: {summary}\n"
        f"세부 위험: {parsed_details}\n\n"
        f"첨부 문서:\n{attachments}"
    )

    try:
        output = call_gemini(system_prompt, user_prompt, temperature=0.3)
    except Exception as exc:
        return {"error": f"Gemini 호출 실패: {exc}"}

    parsed = parse_json(output)
    if parsed:
        return parsed
    return {"raw_output": output}


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
