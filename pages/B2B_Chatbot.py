import streamlit as st
import openai
import google.generativeai as genai
from streamlit_chat import message
import os
import requests
from streamlit_extras.colored_header import colored_header
import pandas as pd

# 페이지 구성 설정
st.set_page_config(layout="wide")

openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "gpt_api_key" not in st.session_state:
    st.session_state.gpt_api_key = openai.api_key # gpt API Key

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = st.secrets["secrets"]["GEMINI_API_KEY"]


# GPT 프롬프트 엔지니어링 함수
def gpt_prompt(user_input):
    base_prompt = f"""
    당신은 전문적인 'SNS 마케팅 콘텐츠'를 자동으로 생성하는 카피라이터입니다.

    **제품 목록:**
    {st.session_state['recommendations']}

    **규칙:**
    1. 한 제품에 대해 매력적인 MMS 마케팅 문구를 작성합니다.
    2. 제품의 주요 특징과 혜택을 강조하고, 소비자의 관심을 끌 수 있도록 합니다.
    3. 문구는 100자 이내로 간결하고 임팩트 있게, 최대한 길게 작성합니다.
    4. 각 문구는 개별적으로 번호를 매겨 구분합니다.
    5. 답변은 존댓말을 사용하며, 브랜드 이미지를 긍정적으로 표현합니다.

    **추가 정보가 필요하다면 사용자에게 요청하세요.**

    **사용자 입력:** {user_input}
    """
    return base_prompt


# GPT 프롬프트 엔지니어링 함수
def gpt_prompt2(user_input):
    base_prompt = f"""
    당신은 전문적인 '가공식품 시장분석 보고서'를 작성하는 애널리스트입니다.

    **분석 대상 제품:**
    {st.session_state['recommendations']}

    **규칙:**
    1. 선택된 가공식품에 대한 시장 분석 보고서를 작성합니다.
    2. 정확한 정보를 기반으로 시장 현황, 경쟁사 분석, 소비자 트렌드 등을 포함합니다.
    3. 보고서는 서론, 본론, 결론의 형식으로 구성합니다.
    4. 데이터는 최신 통계와 사실을 바탕으로 합니다.
    5. 보고서는 전문적이고 객관적인 어조로 작성합니다.

    **추가로 필요한 정보가 있다면 사용자에게 요청하세요.**

    **사용자 입력:** {user_input}
    """
    return base_prompt
    


# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt(user_input):
    # 프롬프트 엔지니어링 관련 로직
    base_prompt = f"""
    당신은 전문적인 'SNS 마케팅 콘텐츠'를 자동으로 생성하는 카피라이터입니다.

    **제품 목록:**
    {st.session_state['recommendations']}

    **규칙:**
    1. 한 제품에 대해 매력적인 MMS 마케팅 문구를 작성합니다.
    2. 제품의 주요 특징과 혜택을 강조하고, 소비자의 관심을 끌 수 있도록 합니다.
    3. 문구는 100자 이내로 간결하고 임팩트 있게, 최대한 길게 작성합니다.
    4. 각 문구는 개별적으로 번호를 매겨 구분합니다.
    5. 답변은 존댓말을 사용하며, 브랜드 이미지를 긍정적으로 표현합니다.

    **추가 정보가 필요하다면 사용자에게 요청하세요.**

    **사용자 입력:** {user_input}
    """
    return base_prompt


# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt2(user_input):
    # 프롬프트 엔지니어링 관련 로직
    base_prompt = f"""
    당신은 전문적인 '가공식품 시장분석 보고서'를 작성하는 애널리스트입니다.

    **분석 대상 제품:**
    {st.session_state['recommendations']}

    **규칙:**
    1. 선택된 가공식품에 대한 시장 분석 보고서를 작성합니다.
    2. 정확한 정보를 기반으로 시장 현황, 경쟁사 분석, 소비자 트렌드 등을 포함합니다.
    3. 보고서는 서론, 본론, 결론의 형식으로 구성합니다.
    4. 데이터는 최신 통계와 사실을 바탕으로 합니다.
    5. 보고서는 전문적이고 객관적인 어조로 작성합니다.

    **추가로 필요한 정보가 있다면 사용자에게 요청하세요.**

    **사용자 입력:** {user_input}
    """
    return base_prompt

# 스트림 표시 함수
def stream_display(response, placeholder):
    text = ''
    for chunk in response:
        if parts := chunk.parts:
            if parts_text := parts[0].text:
                text += parts_text
                placeholder.write(text + "▌")
    return text

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = {
        "gpt": [
            {"role": "system", "content": "안녕하세요, GPT를 기반으로 사용자에게 맞춤형 답변을 드립니다."}
        ],
        "gemini": [
            {"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}
        ]
    }

# 세션 변수 체크
def check_session_vars():
    required_vars = ['selected_gender', 'selected_age']
    for var in required_vars:
        if var not in st.session_state:
            st.warning("필요한 정보가 없습니다. 처음으로 돌아가서 정보를 입력해 주세요.")
            st.stop()

selected_chatbot = st.selectbox(
    "원하는 챗봇을 선택하세요.",
    options=["GPT를 활용한 SNS 마케팅 콘텐츠 자동 생성", "Gemini를 활용한 SNS 마케팅 콘텐츠 자동 생성", "GPT를 활용한 가공식품 시장분석 보고서", "Gemini를 활용한 시장분석 보고서"],
    placeholder="챗봇을 선택하세요.",
    help="선택한 LLM 모델에 따라 다른 챗봇을 제공합니다."
)

if selected_chatbot == "GPT를 활용한 SNS 마케팅 콘텐츠 자동 생성":
    colored_header(
        label='GPT를 활용한 SNS 마케팅 콘텐츠 자동 생성',
        description=None,
        color_name="green-70",
    )

    # 세션 변수 체크
    check_session_vars()

    # 대화 초기화 버튼
    def on_clear_chat_gpt():
        st.session_state.messages["gpt"] = [
            {"role": "system", "content": "안녕하세요, GPT를 기반으로 사용자에게 맞춤형 답변을 드립니다."}
        ]

    st.button("대화 초기화", on_click=on_clear_chat_gpt)

    # 이전 메시지 표시
    if "gpt" not in st.session_state.messages:
        st.session_state.messages["gpt"] = [
            {"role": "system", "content": "안녕하세요, GPT를 기반으로 사용자에게 맞춤형 답변을 드립니다."}
        ]
        
    for msg in st.session_state.messages["gpt"]:
        role = 'user' if msg['role'] == 'user' else 'assistant'
        with st.chat_message(role):
            st.write(msg['content'])

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gpt"].append({"role": "user", "content": prompt})
        with st.chat_message('user'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gpt_prompt(prompt)

        # 모델 호출 및 응답 처리
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o", # gpt-4o-mini-2024-07-18 
                messages=[
                    {"role": "system", "content": enhanced_prompt}
                ] + st.session_state.messages["gpt"],
                max_tokens=1500,
                temperature=0.8,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            text = response.choices[0]['message']['content']

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gpt"].append({"role": "assistant", "content": text})
            with st.chat_message("assistant"):
                st.write(text)
        except Exception as e:
            st.error(f"OpenAI API 요청 중 오류가 발생했습니다: {str(e)}")


elif selected_chatbot == "GPT를 활용한 가공식품 시장분석 보고서":
    colored_header(
        label='GPT를 활용한 가공식품 시장분석 보고서',
        description=None,
        color_name="green-70",
    )

    # 세션 변수 체크
    check_session_vars()

    # 대화 초기화 버튼
    def on_clear_chat_gpt():
        st.session_state.messages["gpt"] = [
            {"role": "system", "content": "안녕하세요, GPT를 기반으로 사용자에게 맞춤형 답변을 드립니다."}
        ]

    st.button("대화 초기화", on_click=on_clear_chat_gpt)

    # 이전 메시지 표시
    if "gpt" not in st.session_state.messages:
        st.session_state.messages["gpt"] = [
            {"role": "system", "content": "안녕하세요, GPT를 기반으로 사용자에게 맞춤형 답변을 드립니다."}
        ]
        
    for msg in st.session_state.messages["gpt"]:
        role = 'user' if msg['role'] == 'user' else 'assistant'
        with st.chat_message(role):
            st.write(msg['content'])

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gpt"].append({"role": "user", "content": prompt})
        with st.chat_message('user'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gpt_prompt2(prompt)

        # 모델 호출 및 응답 처리
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o", # gpt-4o-mini-2024-07-18 
                messages=[
                    {"role": "system", "content": enhanced_prompt}
                ] + st.session_state.messages["gpt"],
                max_tokens=1500,
                temperature=0.8,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            text = response.choices[0]['message']['content']

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gpt"].append({"role": "assistant", "content": text})
            with st.chat_message("assistant"):
                st.write(text)
        except Exception as e:
            st.error(f"OpenAI API 요청 중 오류가 발생했습니다: {str(e)}")

elif selected_chatbot == "Gemini를 활용한 SNS 마케팅 콘텐츠 자동 생성":
    colored_header(
        label='Gemini를 활용한 SNS 마케팅 콘텐츠 자동 생성',
        description=None,
        color_name="blue-70",
    )
    # 세션 변수 체크
    check_session_vars()

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("모델 설정")
        model_name = st.selectbox(
            "모델 선택",
            ['gemini-1.5-flash', "gemini-1.5-pro"]
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
        max_output_tokens = st.number_input("Max Tokens", min_value=1, value=4096, help="생성되는 텍스트의 최대 길이를 제한합니다.")
        top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

    st.button("대화 초기화", on_click=lambda: st.session_state.update({
        "messages": {"gemini": [{"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}]}
    }))

    # 이전 메시지 표시
    if "gemini" not in st.session_state.messages:
        st.session_state.messages["gemini"] = [
            {"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}
        ]
        
    for msg in st.session_state.messages["gemini"]:
        role = 'human' if msg['role'] == 'user' else 'ai'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gemini"].append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gemini_prompt(prompt)

        # 모델 호출 및 응답 처리
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_k": top_k,
                "top_p": top_p
            }
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            chat = model.start_chat(history=st.session_state.messages["gemini"])
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("ai"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "Wait for function calling response..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gemini"].append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")


elif selected_chatbot == "Gemini를 활용한 시장분석 보고서":
    colored_header(
        label='Gemini를 활용한 시장분석 보고서',
        description=None,
        color_name="blue-70",
    )
    # 세션 변수 체크
    check_session_vars()

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("모델 설정")
        model_name = st.selectbox(
            "모델 선택",
            ['gemini-1.5-flash', "gemini-1.5-pro"]
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
        max_output_tokens = st.number_input("Max Tokens", min_value=1, value=4096, help="생성되는 텍스트의 최대 길이를 제한합니다.")
        top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

    st.button("대화 초기화", on_click=lambda: st.session_state.update({
        "messages": {"gemini": [{"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}]}
    }))

    # 이전 메시지 표시
    if "gemini" not in st.session_state.messages:
        st.session_state.messages["gemini"] = [
            {"role": "model", "parts": [{"text": "안녕하세요, Gemini를 기반으로 사용자에게 맞춤형 답변을 드립니다."}]}
        ]
        
    for msg in st.session_state.messages["gemini"]:
        role = 'human' if msg['role'] == 'user' else 'ai'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gemini"].append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gemini_prompt2(prompt)

        # 모델 호출 및 응답 처리
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_k": top_k,
                "top_p": top_p
            }
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            chat = model.start_chat(history=st.session_state.messages["gemini"])
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("ai"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "Wait for function calling response..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gemini"].append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")