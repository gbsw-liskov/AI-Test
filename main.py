import json
import os
from fastapi import APIRouter, FastAPI, Form, UploadFile, File
from typing import Any, List, Optional, Tuple

import google.generativeai as genai

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
        return json.loads(text.strip())
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


# 지원 확장자 -> mime
SUPPORTED_EXTS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

def _ext_of(name: str) -> str:
    name = (name or "").lower().strip()
    for ext in SUPPORTED_EXTS.keys():
        if name.endswith(ext):
            return ext
    return ""

async def build_supported_file_parts(files: Optional[List[UploadFile]]):
    """
    returns: (parts, rejected)
      - parts: Gemini contents에 넣을 파일 파트 리스트
      - rejected: [(filename, reason), ...]
    """
    if not files:
        return [], []

    parts = []
    rejected: List[Tuple[str, str]] = []

    for f in files:
        filename = f.filename or "unnamed"
        ext = _ext_of(filename)

        if not ext:
            rejected.append((filename, "지원 확장자 아님 (pdf/txt/png/jpg/jpeg/docx만 허용)"))
            continue

        data = await f.read()
        mime = SUPPORTED_EXTS[ext]

        # 인라인 bytes로 첨부 (SDK 버전 차이 대비)
        try:
            from google.generativeai import types
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        except Exception:
            # 대안: Files API 업로드로 첨부
            import io
            uploaded = genai.upload_file(io.BytesIO(data), mime_type=mime)
            parts.append(uploaded)

    return parts, rejected

def call_gemini_contents(system_prompt: str, contents, temperature: float = 0.3) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("환경변수 GEMINI_API_KEY가 설정되지 않았습니다.")
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        contents,
        generation_config={"temperature": temperature},
    )
    return response.text or ""


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
    files: Optional[List[UploadFile]] = File(None),
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

    file_parts, rejected = await build_supported_file_parts(files)

    # 원하면: 미지원 파일이 섞이면 아예 400으로 막아도 됨
    # if rejected:
    #     return {"error": "미지원 파일이 포함되어 있습니다.", "rejected": rejected}

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

    # ✅ 텍스트 + 파일을 함께 contents로 전달
    contents = [
        "다음 매물 정보를 검토하고 위험도를 산출해라. 첨부 파일 내용도 근거로 활용하라.",
        property_block,
    ]
    if file_parts:
        contents.append("첨부 파일들:")
        contents.extend(file_parts)
    else:
        contents.append("첨부 파일 없음")

    try:
        output = call_gemini_contents(system_prompt, contents, temperature=0.2)
    except Exception as exc:
        return {"error": f"Gemini 호출 실패: {exc}"}

    parsed = parse_json(output)
    if parsed:
        # 미지원 파일 목록은 참고용으로 같이 내려줄 수도 있음
        if rejected:
            parsed["rejectedFiles"] = [{"filename": fn, "reason": rsn} for fn, rsn in rejected]
        return parsed

    return {"raw_output": output, "rejectedFiles": [{"filename": fn, "reason": rsn} for fn, rsn in rejected]}




app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
