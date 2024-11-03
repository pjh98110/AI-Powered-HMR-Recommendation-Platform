import streamlit as st
from st_pages import Page, show_pages
import pandas as pd
import numpy as np
import random
import os
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header
import requests

# from surprise import Dataset, Reader, SVD, NMF
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# import seaborn as sns
# import matplotlib.pyplot as plt
# import shap
# import time

# 페이지 구성 설정
st.set_page_config(
    page_title="생성형 AI 기반 간편식(HMR) 추천 플랫폼",
    page_icon="🏠",
    layout="wide",
)

# Streamlit의 경우 로컬 환경에서 실행할 경우 터미널 --> Streamlit run "파일 경로/파일명.py"로 실행 / 로컬 환경과 스트리밋 웹앱 환경에서 기능의 차이가 일부 있을 수 있음
# 파일 경로를 잘못 설정할 경우 오류가 발생하고 실행이 불가능하므로 파일 경로 수정 필수
# 데이터 파일의 경우 배포된 웹앱 깃허브에서 다운로드 가능함
# 비공개 데이터 때문에 로컬에서는 정상 작동하지만, 배포된 코드에는 기능 제한과 추천시스템 구조만 보여줌.


# 배경 색상 설정
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFAF1; 
    }
    </style>
    """,
    unsafe_allow_html=True
)


show_pages(
    [
        Page("Home.py", "생성형 AI 기반 간편식(HMR) 추천 플랫폼", "🏠"),
        Page("pages/B2C_Chatbot.py", "B2C 간편식 추천 챗봇", "🛒"),
        Page("pages/B2B_Chatbot.py", "B2B 간편식 대시보드 챗봇", "🏢"),
        # Page("pages/Tableau.py", "Tableau", "🗺️"),
        Page("pages/Explainable_AI.py", "Explainable_AI", "📑"),
    ]
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

DATA_PATH = "./"

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# X = load_csv(f'{DATA_PATH}retail_data.csv')

# # 추천 시스템에 사용할 모델 학습 및 데이터 준비
# # Reader 설정
# reader = Reader(rating_scale=(1, 10))

# # Surprise Dataset 로드
# data = Dataset.load_from_df(X[['user_id', 'item_id', 'rating']], reader)

# # 전체 학습셋 생성
# trainset = data.build_full_trainset()

# # 협업 필터링 모델 학습 (SVD)
# algo = SVD()
# algo.fit(trainset)

# # TF-IDF 벡터화 및 콘텐츠 기반 유사도 계산
# tfidf = TfidfVectorizer()
# tfidf_matrix = tfidf.fit_transform(X['M3_상품명'])
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# indices = pd.Series(X.index, index=X['M3_상품명']).drop_duplicates()

# # NMF 잠재 요인 협업 필터링링 학습
# algo_nmf = NMF()
# algo_nmf.fit(trainset)



def reset_seeds(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

reset_seeds(42)

# # 한글 폰트 설정 함수
# def set_korean_font():
#     font_path = f"{DATA_PATH}NanumGothic.ttf"  # 폰트 파일 경로

#     from matplotlib import font_manager, rc
#     font_manager.fontManager.addfont(font_path)
#     rc('font', family='NanumGothic')

# # 한글 폰트 설정 적용
# set_korean_font()


# 세션 변수에 저장
if 'type_of_case' not in st.session_state:
    st.session_state.type_of_case = None

if 'selected_gender' not in st.session_state:
    st.session_state.selected_gender = "여성"

if 'selected_age' not in st.session_state:
    st.session_state.selected_age = "40~49세"

if 'selected_taste' not in st.session_state:
    st.session_state.selected_taste = "달콤한 맛"

if 'selected_allergy' not in st.session_state:
    st.session_state.selected_allergy = "알레르기 없음"

if 'selected_time' not in st.session_state:
    st.session_state.selected_time = "점심"



# # 협업 필터링 추천 함수
# def recommend_collaborative(user_id, num_recommendations=5):
#     user_rated_items = X[X['user_id'] == user_id]['item_id'].unique()
#     all_items = X['item_id'].unique()
#     items_to_predict = [iid for iid in all_items if iid not in user_rated_items]
    
#     predictions = [algo.predict(user_id, iid) for iid in items_to_predict]
#     predictions.sort(key=lambda x: x.est, reverse=True)
#     top_n = predictions[:num_recommendations]
#     recommended_items = [X[X['item_id'] == pred.iid]['M3_상품명'].iloc[0] for pred in top_n]
#     return recommended_items

# # 콘텐츠 기반 필터링 추천 함수
# def recommend_content_based(item_name, num_recommendations=5):
#     if item_name not in indices:
#         st.write(f"'{item_name}' 상품명을 찾을 수 없습니다.")
#         return []
#     idx = indices[item_name]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     # Exclude the first item since it's the item itself
#     sim_scores = sim_scores[1:num_recommendations+1]
#     item_indices = [i[0] for i in sim_scores]
#     recommended_items = X['M3_상품명'].iloc[item_indices].unique()
#     return recommended_items.tolist()

# # 하이브리드 추천 함수
# def recommend_hybrid(user_id, num_recommendations=5, alpha=0.5):
#     collab_recommendations = recommend_collaborative(user_id, num_recommendations*2)
#     last_items = X[X['user_id'] == user_id]['M3_상품명']
#     if last_items.empty:
#         st.write("해당 사용자의 구매 이력이 없습니다.")
#         return []
#     last_item = last_items.iloc[-1]
#     content_recommendations = recommend_content_based(last_item, num_recommendations*2)
    
#     # Combine and rank recommendations
#     combined = collab_recommendations + content_recommendations
#     combined_counts = pd.Series(combined).value_counts()
#     recommended_items = combined_counts.head(num_recommendations).index.tolist()
#     return recommended_items

# # 도메인 기반 추천 함수
# def recommend_knowledge_based(gender, age_group, num_recommendations=5):
#     filtered_items = X[(X['P1_성별'] == gender) & (X['P2_연령대'] == age_group)]
#     popular_items = filtered_items.groupby('M3_상품명').size().reset_index(name='counts')
#     popular_items = popular_items.sort_values('counts', ascending=False)
#     recommended_items = popular_items['M3_상품명'].head(num_recommendations).unique()
#     return recommended_items

# # 모델 기반 추천 함수 (NMF 사용)
# def recommend_model_based(user_id, num_recommendations=5):
#     user_rated_items = X[X['user_id'] == user_id]['item_id'].unique()
#     all_items = X['item_id'].unique()
#     items_to_predict = [iid for iid in all_items if iid not in user_rated_items]
    
#     predictions = [algo_nmf.predict(user_id, iid) for iid in items_to_predict]
#     predictions.sort(key=lambda x: x.est, reverse=True)
#     top_n = predictions[:num_recommendations]
#     recommended_items = [X[X['item_id'] == pred.iid]['M3_상품명'].iloc[0] for pred in top_n]
#     return recommended_items


def recommend_random_products(num_recommendations=5):
    unique_products = ['[사미헌] 맑은곰탕 500g', '[키친델리] 어메이징 얼큰 곱창부속전골 1320g',
       '[CJ제일제당] 햇반 컵반 순두부찌개국밥 173g', '[CJ제일제당] 햇반 컵반 황태국밥 170g',
       '[CJ제일제당] 햇반 컵반 미역국밥 167g', '[비비고] 삼계탕 800g', '[비비고] 사골곰탕 500g',
       '[비비고] 육개장 500g', '[비비고] 두부김치찌개 460g', '[비비고] 된장찌개 460g',
       '[비비고] 두부듬뿍 된장찌개 460g', '[비비고] 스팸 부대찌개 460g', '[비비고] 닭곰탕 500g',
       '[비비고] 설렁탕 500g', '[비비고] 소고기미역국 500g', '[비비고] 소고기무국 500g',
       '[비비고] 소고기장터국 500g', '[비비고] 콩나물황태국 500g', '[비비고] 사골곰탕 300g',
       '[비비고] 갈비탕 400g', '[비비고] 차돌된장찌개 460g', '[비비고] 돼지고기 김치찌개 460g',
       '[비비고] 남도식추어탕 460g', '[비비고] 시래기 감자탕 460g', '[비비고] 사골순댓국 460g',
       '[비비고] 한우사골곰탕 500g', '[비비고] 도가니곰탕 460g', '[비비고] 꼬리곰탕 460g',
       '[비비고] 누룽지 닭다리삼계탕 600g', '[비비고] 사골 시래기된장국 460g',
       '[비비고] 진국육수 소고기양지 500g', '[비비고] 진국육수 멸치해물 500g',
       '[농심] 한일관 서울식 우거지 갈비탕 460g', '[농심] 한일관 서울식 된장찌개 460g',
       '[오뚜기] 대전식 쇠고기무국 500g', '[오뚜기] 청주식 돼지김치짜글이 450g',
       '[오뚜기] 부산식 얼큰돼지국밥 500g', '[오뚜기] 산청식 우렁된장국 500g',
       '[오뚜기] 옛날 사골곰탕 500g', '[오뚜기] 안동식 쇠고기국밥 500g', '[오뚜기] 서울식 설렁탕 500g',
       '[오뚜기] 종로식 도가니탕 500g', '[오뚜기] 수원식 우거지갈비탕 500g',
       '[오뚜기] 남도식 한우미역국 500g', '[오뚜기] 옛날 삼계탕 900g', '[오뚜기] 부산식 돼지국밥 500g',
       '[오뚜기] 의정부식 부대찌개 500g', '[오뚜기] 옛날 육개장 500g', '[오뚜기] 옛날 사골곰탕 350g',
       '[오뚜기] 옛날 육개장 300g', '[오뚜기] 광주식 애호박 고추장찌개 450g',
       '[오뚜기] 나주식 쇠고기곰탕 500g', '[오뚜기] 병천식 얼큰순대국밥 500g',
       '[오뚜기] 양평식 선지해장국 500g', '[오뚜기] 대구식 쇠고기 육개장 500g',
       '[오뚜기] 마포식 차돌된장찌개 500g', '[동원] 양반 소고기미역국 460g',
       '[동원] 양반 수라 통다리 삼계탕 460g', '[동원] 양반 수라 부대찌개 460g',
       '[동원] 양반 수라 돼지고기김치찌개 460g', '[동원] 양반 나주식 곰탕 460g',
       '[동원] 양반 수라 두배진한 사골곰탕 460g', '[동원] 양반 수라 완도전복 육개장 460g',
       '[동원] 양반 수라 통다리 닭볶음탕 490g', '[동원] 양반 수라 보양 추어탕 460g',
       '[동원] 양반 수라 차돌된장찌개 460g', '[동원] 양반 수라 도가니설렁탕 460g',
       '[동원] 양반 차돌육개장 460g', '[동원] 양반 백합 미역국 460g',
       '[동원] 양반 수라 왕갈비탕 460g', '[동원] 양반 수라 한우사골곰탕 460g',
       '[동원] 양반 진국 소고기무국 460g', '[동원] 양반 진국 사골곰탕 500g',
       '[동원] 양반 진국 사골곰탕 300g', '[동원] 양반 김치 청국장찌개 460g',
       '[동원] 양반 백합우거지된장국 460g', '[동원] 양반 한우사골 시래기국 460g',
       '[동원] 양반 수라 우거지감자탕 460g', '[동원] 양반 수라 차돌육개장 460g',
       '[동원] 양반 리챔 부대전골 키트 870g', '[청정원] 호밍스 남도추어탕 450g',
       '[청정원] 호밍스 사골곰탕 300g', '[청정원] 호밍스 사골 시래기된장국 450g',
       '[청정원] 호밍스 얼큰 알탕 450g', '[청정원] 호밍스 사골 선지해장국 450g',
       '[청정원] 호밍스 얼큰 순두부찌개 450g', '[청정원] 호밍스 한우곰탕 300g',
       '[청정원] 호밍스 한우 진곰탕 450g', '[청정원] 호밍스 한우진곰탕 450g',
       '[청정원] 호밍스 나주곰탕 450g', '[청정원] 호밍스 소머리 곰탕 450g',
       '[청정원] 호밍스 도가니탕 450g', '[청정원] 호밍스 소고기미역국 450g',
       '[청정원] 호밍스 파육개장 500g', '[청정원] 호밍스 부산식 곱창전골 760g',
       '[청정원] 호밍스 얼큰 차돌 육개장 450g', '[청정원] 호밍스 김치 콩나물국 450g',
       '[청정원] 호밍스 깻잎 곱창전골 400g', '[청정원] 호밍스 고깃집 차돌 된장찌개 450g',
       '[청정원] 호밍스 사골 진곰탕 500g', '[청정원] 호밍스 낙곱새 전골 800g',
       '[청정원] 호밍스 얼큰 닭개장 450g', '[청정원] 호밍스 맑은 닭곰탕 450g',
       '[청정원] 호밍스 얼큰 고기 짬뽕탕 450g', '[청정원] 호밍스 사천식마라탕 450g',
       '[청정원] 호밍스 얼큰 김치 만두전골 680g', '[노브랜드] 꽉찬 종합어묵 1000g',
       '[대림선] 대림선어묵 국탕종합 340g', '[풀무원] 두부요리킷 정통 순두부찌개 602g',
       '[풀무원] 두부요리킷 얼큰 순두부찌개 600g', '[풀무원] 두부요리킷 바지락 순두부찌개 602g',
       '[풀무원] 반듯한식 진한 사골곰탕 500g', '[요리하다] 황태해장국 500g',
       '[피코크] 금돼지식당 김치찌개 500g', '[삼호어묵] 정통어묵탕 336g',
       '[비비고] 두부 청국장찌개 460g', '[비비고] 양지설렁탕 700g', '[비비고] 양지곰탕 700g',
       '[비비고] 본갈비탕 700g', '[비비고] 수삼갈비탕 400g', '[비비고] 소고기 듬뿍 설렁탕 460g',
       '[비비고] 소고기 듬뿍 육개장 460g', '[비비고] 소고기 듬뿍 미역국 460g',
       '[비비고] 양지곰탕 700g (350g x 2)', '[비비고] 본갈비탕 700g (350g x 2)',
       '[쿡킷] 모둠사리 스팸 부대전골 690g', '[비비고] 고기순대국 700g (350g x 2)',
       '[비비고] 고기순대국 700g', '[비비고] 순살 감자탕 700g', '[쿡킷] 진한육수 곱창전골 760g',
       '[이음식] 양평해장국 800g', '[교동] 육개장 500g', '[교동] 사골우거지국 500g',
       '[교동] 오징어무국 500g', '[피코크] 진한 시골 장터국 500g', '[피코크] 진한 순살 감자탕 500g',
       '[홈플러스시그니처] 푸짐한 스팸 부대찌개 1120g', '[홈플러스시그니처] 나혼자 푸짐한 스팸 부대찌개 525g',
       '[피코크] 푸짐한 대구 매운탕 1013g', '[심플리쿡] 햄폭탄 부대전골 818g',
       '[삼진어묵] 딱 한끼 어묵탕 순한맛 308g', '[요리하다] 육개장 500g',
       '[오아시스] 우리한우 진한곰탕 600g', '[요리하다] 사골곰탕 500g', '[쟌슨빌] 더진한 부대찌개 500g',
       '[홈플러스시그니처] 소불고기 버섯전골 490g', '[홈플러스시그니처] 우삼겹 된장찌개 480g',
       '[홈플러스시그니처] 소불고기 버섯전골 300g', '[미스타셰프] 육개장 600g',
       '[미스타셰프] 부대찌개 600g', '[미가인] 의정부식 부대찌개 750g',
       '[홈플러스시그니처] 쟌슨빌 부대찌개 985g', '[홈플러스시그니처] 감자수제비 순두부찌개 1100g',
       '[미가인] 본질에 충실한 부대찌개 700g', '[홈플러스시그니처] 이건꼭사야해 부대대찌개 2.5kg',
       '[노브랜드] 사골육수 500g', '[홈플러스시그니처] 사골곰탕 500g',
       '[노브랜드] 쇠고기 사골 미역국 500g', '[남가네] 설악추어탕 450g',
       '[아빠식당] 푸짐한 곱창전골 800g', '[궁] 왕 갈비탕 1000g',
       '[피코크] 쟌슨빌 소시지 부대찌개 500g', '[강창구찹쌀진순대] 찹쌀 진순대국 600g',
       '[놀부] 아빠식당 부대찌개 600g', '[외갓집] 진심 육개장 600g', '[외갓집] 진심 갈비탕 650g',
       '[신의주찹쌀순대] 신의주 찹쌀순대국 600g', '[해화당] 뼈없는 갈비탕 정 900g',
       '[배민이지] 얼큰한 국물 순대가 듬뿍 순대국 700g', '[오프라이스] 한촌설렁탕 사골곰탕 500g',
       '[프레시지] 서울식 불고기 전골 424.5g', '[프레시지] 밀푀유 나베 850g',
       '[프레시지] 더큰 햄가득 부대전골 868g', '[마이셰프] 밀푀유나베 & 칼국수 1129g',
       '[피코크] 우리집 콩비지찌개 500g', '[아워홈] 시원한 황태 해장국 300g',
       '[피코크] 샤브샤브 요리재료 870g', '[피코크] 순두부찌개 요리재료 804g',
       '[피코크] 된장찌개 요리재료 780g', '[피코크] 강릉식 짬뽕순두부 1010g',
       '[피코크] 영월식 청국장 930g', '[피코크] 어메이징 부대찌개 1252g',
       '[피코크] 정갈한 쇠고기 미역국 500g', '[피코크] 정갈한 쇠고기무국 500g',
       '[피코크] 우리집 차돌 된장찌개 500g', '[피코크] 정갈한 오징어무국 500g',
       '[피코크] 정갈한 콩나물김칫국 500g', '[추추] 추어탕 500g', '[푸드어셈블] 채선당 샤브샤브 955g',
       '[채선당] 샤브샤브 845g', '[안원당] 우거지 감자탕 920g', '[더오담] 콩비지찌개 500g',
       '[신의주찹쌀순대] 신의주 얼큰순대국 600g', '[진실된손맛] 한우사골 양지곰탕 500g',
       '[피코크] 송탄식 부대찌개 738g', '[피코크] 의정부식 부대찌개 680g', '[노브랜드] 곱창전골 400g',
       '[더미식] 사골곰탕 500g', '[더미식] 설렁탕 350g', '[더미식] 소고기미역국 350g',
       '[더미식] 닭개장 350g', '[더미식] 부대찌개 350g', '[더미식] 소고기육개장 350g',
       '[더미식] 차돌육개장 350g', '[더미식] 우렁된장찌개 350g', '[더미식] 시래기된장국 350g',
       '[피코크] 전주식 콩나물해장국 500g', '[노브랜드] 꼬치어묵 518g', '[노브랜드] 매운 꼬치어묵 528g',
       '[어나더테이블] 고래사어묵으로 만든 김치우동전골 650g', '[프레시지] 캠핑포차 김치어묵 우동전골 1080g',
       '[프레시지] 북창동 소고기 순두부 찌개 620g', '[피코크] 밀푀유 나베 845g',
       '[피코크] 소불고기 전골 444.5g', '[피코크] 깊고 진한 버섯어묵 전골 611g',
       '[피코크] 리북방 순대전골 1.1Kg', '[노브랜드] 파개장 500g',
       '[피코크] 어랑손만두 만두전골 1.15kg', '[피코크] 무교동식 북엇국 500g',
       '[바른식] 부산 조방낙지 낙곱새 700g', '[바른식] 강릉식 짬뽕순두부 찌개 860g',
       '[바른식] 등촌식 미나리 샤브전골 845g', '[홍익궁중전통] 육개장 750g',
       '[그리팅] 한우우거지탕 800g', "[99's fresh] 소고기 버섯 전골 390g",
       '[사미헌] 갈비탕 1kg', '[하루한킷] 송탄식 부대찌개 1058g', '[곰곰] 갈비탕 600g',
       '[곰곰] 더 오리지널 부대찌개 1kg', '[통뼈] 통뼈 뼈해장국 1.8kg (900g x 2)']
    if len(unique_products) < num_recommendations:
        num_recommendations = len(unique_products)
    recommended_items = random.sample(unique_products, num_recommendations)
    return recommended_items


# # 통계치 계산 및 저장 함수
# @st.cache_data
# def calculate_statistics():
#     # 평균 평점 계산
#     product_stats = X.groupby('M3_상품명')['rating'].mean().reset_index(name='평균 평점')
#     # 성별에 따른 평균 평점
#     gender_stats = X.groupby('P1_성별')['rating'].mean().reset_index(name='평균 평점')
#     # 연령대에 따른 평균 평점
#     age_stats = X.groupby('P2_연령대')['rating'].mean().reset_index(name='평균 평점')
#     return product_stats, gender_stats, age_stats

# product_stats, gender_stats, age_stats = calculate_statistics()



# 타이틀
colored_header(
    label= '생성형 AI 기반 간편식(HMR) 추천 플랫폼',
    description=None,
    color_name="blue-70",
)



# [사이드바]
st.sidebar.markdown(f"""
            <span style='font-size: 20px;'>
            <div style=" color: #000000;">
                <strong>추천시스템 및 정보 입력</strong>
            </div>
            """, unsafe_allow_html=True)


# 사이드바에서 사용자 정보 입력
selected_gender = st.sidebar.selectbox(
    "(1) 당신의 성별을 선택하세요.",
    ["남성", "여성"]
)
st.session_state.selected_gender = selected_gender

selected_age = st.sidebar.selectbox(
    "(2) 당신의 연령대를 선택하세요.",
    ["25~29세", "30~39세", "40~49세", "50~59세", "60~69세"]
)
st.session_state.selected_age = selected_age

selected_taste = st.sidebar.selectbox(
    "(3) 선호하는 맛을 선택하세요.",
    ["달콤한 맛", "짠 맛", "매운맛", "신 맛", "고소한 맛"]
)
st.session_state.selected_taste = selected_taste

selected_allergy = st.sidebar.text_input(
    "(4) 보유한 알레르기를 알려주세요.",
    placeholder = "알레르기 없음, 땅콩 알레르기 등"
)
st.session_state.selected_allergy = selected_allergy

selected_time = st.sidebar.selectbox(
    "(5) 간편식을 먹는 시간대를 알려주세요.",
    ["아침", "점심", "저녁", "야식"]
)
st.session_state.selected_time = selected_time


definitions = {
    '사용자 협업 필터링': '사용자 기반 협업 필터링: 유저 간의 유사도가 높을 수록 높은 가중치를 부여하는 방식으로, 특정 유저가 아직 구매하지 않았으나 동질 그룹의 다른 유저가 선호하는 아이템을 추천',
    '콘텐츠 기반 필터링': '콘텐츠 기반 필터링: 아이템의 특성과 사용자의 선호도를 분석하여 비슷한 아이템 추천하는 방식',
    '하이브리드 추천 시스템': '하이브리드 추천시스템(협업 필터링+콘텐츠 기반 필터링): 컨텐츠 기반 추천시스템과 협업 필터링을 결합한 모델. 넷플릭스는 협업 필터링을 사용해 유사한 사용자 간의 시청/검색 기록을 비교하고, 콘텐츠 기반 필터링을 사용해 사용자가 높게 평가한 영화의 특징을 공유하는 영화를 제공',
    '도메인 기반 필터링': '도메인 기반 필터링: 추천하고자 하는 분야의 도메인 지식을 활용해 추천하는 방식',
    '잠재 요인 협업 필터링': '잠재 요인 협업 필터링: 사용자와 아이템 간의 평점 행렬 속에 숨어 있는 잠재 요인 행렬을 추출하여 내적 곱을 통해 사용자가 평가하지 않은 항목들에 대한 평점까지 예측하여 추천하는 방법',
    '앙상블 추천시스템': '앙상블 추천시스템: 여러 개의 추천 알고리즘을 결합하여 더 정확하고 신뢰할 수 있는 추천을 제공하는 방법. Hard Voting을 사용하여 각 모델의 추천 결과를 취합하고, 가장 많이 추천된 아이템을 선택합니다.'
}

image_names = {
    '사용자 협업 필터링': 'User',
    '콘텐츠 기반 필터링': 'Content',
    '하이브리드 추천 시스템': 'Hybrid',
    '도메인 기반 필터링': 'Domain', 
    '잠재 요인 협업 필터링': 'Factor',
    '앙상블 추천시스템': 'Voting'
}

# Streamlit App
options = ['사용자 협업 필터링', '콘텐츠 기반 필터링', '하이브리드 추천 시스템', '도메인 기반 필터링', '잠재 요인 협업 필터링', '앙상블 추천시스템']

selected_option = st.selectbox(
    "사용할 추천시스템을 선택하세요.",
    options=options, 
    placeholder="추천시스템 하나를 선택하세요.",
    help="선택한 추천시스템에 따라 다른 결과를 제공합니다."
)

st.session_state.selected_option = selected_option

# '앙상블 추천시스템' 선택 시 멀티 선택 박스 표시 및 최소 선택 조건 설정
if selected_option == '앙상블 추천시스템':
    ensemble_options = [option for option in options if option != '앙상블 추천시스템']
    selected_models = st.multiselect(
        "앙상블에 포함할 추천시스템을 선택하세요 (최소 2개 이상)",
        options=ensemble_options,
        help="사용할 추천시스템을 여러 개 선택할 수 있습니다. 최소 2개를 선택해야 합니다."
    )

    # 선택된 모델이 2개 미만인 경우 경고 메시지 표시
    if len(selected_models) < 2:
        st.warning("최소 2개 이상의 추천시스템을 선택하세요.")
    else:
        st.write(definitions['앙상블 추천시스템'])
        image_path = f"{DATA_PATH}{image_names['앙상블 추천시스템']}.png"
        st.image(image_path)
        
        # 추천 버튼이 나타나고 클릭 시 추천 진행
        if st.button("추천받기"):
            recommendations = recommend_random_products()
            st.session_state['recommendations'] = recommendations
            # 추천 결과 출력
            st.write("### 추천 상품 목록:")
            for item in recommendations:
                st.write(f"- {item}")

else:
    # 추천 버튼 클릭 시 동작 (앙상블 외의 다른 추천시스템)
    if st.button("😊 추천받기"):
        st.markdown(""" **생성형 AI 기반 추가 정보를 원하시면 아래 :blue[챗봇]을 클릭해주세요.** """)
        
        # 추천 시스템에 따라 추천 처리
        if selected_option in definitions:
            st.write(definitions[selected_option])
            image_path = f"{DATA_PATH}{image_names[selected_option]}.png"
            st.image(image_path)
            recommendations = recommend_random_products()
            st.session_state['recommendations'] = recommendations
        else:
            recommendations = []
        
        # 추천 결과 출력
        st.write("### 추천 상품 목록:")
        if len(recommendations) > 0:
            for item in recommendations:
                st.write(f"- {item}")
        else:
            st.write("추천 결과가 없습니다.")

# 페이지 전환 버튼 스타일 설정
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #FED556;
        color: #000000; /* 텍스트 색상 */
        width: 100%;
        display: inline-block;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 페이지 전환 함수 정의
def page1():
    want_to_B2C_Chatbot = st.button("🛒 B2C 간편식 추천 챗봇")
    if want_to_B2C_Chatbot:
        st.session_state.type_of_case = "B2C 간편식 추천 챗봇"
        switch_page("B2C 간편식 추천 챗봇")
        
def page2():
    want_to_B2B_Chatbot = st.button("🏢 B2B 간편식 대시보드 챗봇")
    if want_to_B2B_Chatbot:
        st.session_state.type_of_case = "B2B 간편식 대시보드 챗봇"
        switch_page("B2B 간편식 대시보드 챗봇")

# def page3():
#     want_to_Tableau = st.button("Tableau")
#     if want_to_Tableau:
#         st.session_state.type_of_case = "Tableau"
#         switch_page("Tableau")

def page3():
    want_to_Explainable_AI = st.button("📑 Explainable_AI")
    if want_to_Explainable_AI:
        st.session_state.type_of_case = "Explainable_AI"
        switch_page("Explainable_AI")

# 페이지 전환 버튼 배치
col1, col2, col3 = st.columns(3)
with col1:
    page1()
with col2:
    page2()
with col3:
    page3()



st.sidebar.markdown(f"""
            <span style='font-size: 20px;'>
            <div style=" color: #000000;">
                <strong> 간편식 관련 추가정보 </strong>
            </div>
            """, unsafe_allow_html=True)

# 식품(첨가물)품목제조보고 컬럼 매핑 (영어 키 -> 한국어 설명)
column_mapping = {
    'LCNS_NO': '인허가번호',
    'BSSH_NM': '업소명',
    'PRDLST_REPORT_NO': '품목제조번호',
    'PRMS_DT': '허가일자',
    'PRDLST_NM': '제품명',
    'PRDLST_DCNM': '품목유형명',
    'PRODUCTION': '생산종료여부',
    'HIENG_LNTRT_DVS_NM': '고열량저영양식품여부',
    'CHILD_CRTFC_YN': '어린이기호식품품질인증여부',
    'POG_DAYCNT': '소비기한',
    'LAST_UPDT_DTM': '최종수정일자',
    'INDUTY_CD_NM': '업종',
    'QLITY_MNTNC_TMLMT_DAYCNT': '품질유지기한일수',
    'USAGE': '용법',
    'PRPOS': '용도',
    'DISPOS': '제품형태',
    'FRMLC_MTRQLT': '포장재질'
}

# 조리식품 레시피 컬럼 매핑 (영어 키 -> 한국어 설명)
recipe_column_mapping = {
    'RCP_SEQ': '일련번호',
    'RCP_NM': '메뉴명',
    'RCP_WAY2': '조리방법',
    'RCP_PAT2': '요리종류',
    'INFO_WGT': '중량(1인분)',
    'INFO_ENG': '열량',
    'INFO_CAR': '탄수화물',
    'INFO_PRO': '단백질',
    'INFO_FAT': '지방',
    'INFO_NA': '나트륨',
    'HASH_TAG': '해쉬태그',
    'ATT_FILE_NO_MAIN': '이미지경로(소)',
    'ATT_FILE_NO_MK': '이미지경로(대)',
    'RCP_PARTS_DTLS': '재료정보',
    'MANUAL01': '만드는법_01',
    'MANUAL_IMG01': '만드는법_이미지_01',
    'MANUAL02': '만드는법_02',
    'MANUAL_IMG02': '만드는법_이미지_02',
    'MANUAL03': '만드는법_03',
    'MANUAL_IMG03': '만드는법_이미지_03',
    'MANUAL04': '만드는법_04',
    'MANUAL_IMG04': '만드는법_이미지_04',
    'MANUAL05': '만드는법_05',
    'MANUAL_IMG05': '만드는법_이미지_05',
    'MANUAL06': '만드는법_06',
    'MANUAL_IMG06': '만드는법_이미지_06',
    'MANUAL07': '만드는법_07',
    'MANUAL_IMG07': '만드는법_이미지_07',
    'MANUAL08': '만드는법_08',
    'MANUAL_IMG08': '만드는법_이미지_08',
    'MANUAL09': '만드는법_09',
    'MANUAL_IMG09': '만드는법_이미지_09',
    'MANUAL10': '만드는법_10',
    'MANUAL_IMG10': '만드는법_이미지_10',
    'RCP_NA_TIP': '저감 조리법 TIP'
}

# 추천 버튼 클릭 시 동작 COOKRCP01
selected_api = st.sidebar.selectbox(
    "원하는 추가 정보를 API로 제공해드립니다.",
    ["식품(첨가물)품목제조보고", "조리식품 레시피"]
)
if st.sidebar.button("📝 데이터 불러오기"):
    if selected_api == "식품(첨가물)품목제조보고":
        # API 기본 정보 설정
        API_KEY = st.secrets["secrets"]["FOOD_API"]
        SERVICE_ID = 'I1250'  # 서비스명
        DATA_TYPE = 'json'  # 요청 파일 타입 (json 또는 xml)
        START_IDX = '1'  # 요청 시작 위치
        END_IDX = '100'  # 요청 종료 위치
        BASE_URL = f'http://openapi.foodsafetykorea.go.kr/api/{API_KEY}/{SERVICE_ID}/{DATA_TYPE}/{START_IDX}/{END_IDX}'

        # API 요청
        response = requests.get(BASE_URL)

        # 응답 확인 및 데이터 출력
        if response.status_code == 200:
            data = response.json()  # 데이터 파싱
            
            # "row" 키의 경로 확인 (실제 응답 데이터 구조에 따라 수정 필요)
            rows = data.get(SERVICE_ID, {}).get("row", [])
            
            # 결과가 있으면 처리
            if rows:
                # DataFrame 생성 및 컬럼 매핑
                df = pd.DataFrame(rows)
                
                # 컬럼 매핑 및 깔끔하게 원하는 컬럼만 표시
                df = df.rename(columns=column_mapping)
                display_columns = list(column_mapping.values())
                df_display = df[display_columns]

                st.dataframe(df_display)

            else:
                st.write("데이터가 없습니다. 응답 데이터 구조를 확인하세요.")
        else:
            st.write(f"API 요청 오류: {response.status_code}")

    elif selected_api == "조리식품 레시피":
        # API 기본 정보 설정
        API_KEY = st.secrets["secrets"]["FOOD_API"]
        SERVICE_ID = 'COOKRCP01'  # 서비스명
        DATA_TYPE = 'json'  # 요청 파일 타입 (json 또는 xml)
        START_IDX = '1'  # 요청 시작 위치
        END_IDX = '100'  # 요청 종료 위치
        BASE_URL = f'http://openapi.foodsafetykorea.go.kr/api/{API_KEY}/{SERVICE_ID}/{DATA_TYPE}/{START_IDX}/{END_IDX}'

        # API 요청
        response = requests.get(BASE_URL)

        # 응답 확인 및 데이터 출력
        if response.status_code == 200:
            data = response.json()  # 데이터 파싱
            
            # "row" 키의 경로 확인 (실제 응답 데이터 구조에 따라 수정 필요)
            rows = data.get(SERVICE_ID, {}).get("row", [])
            
            # 결과가 있으면 처리
            if rows:
                # DataFrame 생성 및 컬럼 매핑
                df = pd.DataFrame(rows)
                
                # 컬럼 매핑 및 필요한 컬럼만 표시
                df = df.rename(columns=recipe_column_mapping)

                # 필요한 컬럼만 표시 (실제 필요한 컬럼들로 구성)
                display_columns = [
                    '일련번호', '메뉴명', '조리방법', '요리종류', '열량', '탄수화물', '단백질', '지방', '나트륨', '재료정보', 
                    '만드는법_01', '만드는법_02', '만드는법_03', '만드는법_04', '만드는법_05', '만드는법_06', '만드는법_07', '이미지경로(소)', '이미지경로(대)', '저감 조리법 TIP'
                ]
                df_display = df[display_columns]

                # Streamlit에 데이터프레임 출력
                st.dataframe(df_display)

            else:
                st.write("데이터가 없습니다. 응답 데이터 구조를 확인하세요.")
        else:
            st.write(f"API 요청 오류: {response.status_code}")



st.sidebar.link_button("🚛 유통데이터 서비스 플랫폼", "https://m.retaildb.or.kr/")
st.sidebar.link_button("🚛 유통상품 표준DB", "https://www.allproductkorea.or.kr/")
