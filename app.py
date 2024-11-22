import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tiktoken

#OpenAI API키 설정
#.env파일에서 api키 가져오기
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()
# API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")


#
#Streamlit 앱
#타이틀
st.title("AI 창업 코치: TopDog(탑독)")
#헤더
st.header(":hatching_chick: Beta.ver")
#Topdog 알아보기 버튼
if st.button(":eyes: Topdog 알아보기"):
    st.write("""
    "Topdog"이라는 말은 "underdog effect"라는 표현에서 가져온 이름입니다.\n
    "underdog effect"라는 표현이 있습니다. 해당 표현에서 "underdog"은 약자를 뜻하는 단어로 사용되고 있는데요. \n
    창업씬에서 약자에 속하는 예비/초기 단계의 창업자들이 "underdog"이 아닌 "TopDog"이 되어 유쾌한 반란을 일으키시면 좋겠습니다.
    """)
#서브 타이틀
st.markdown("### :sparkles: AI 코치에게 창업에 대한 조언을 구해보세요! :sparkles:")

# 데이터 폴더와 PDF 파일 리스트
data_folder = "data"
pdf_files = [
    "learn_1.pdf", "learn_2.pdf", "learn_3.pdf", "learn_4.pdf",
    "learn_5.pdf", "pass_1.pdf", "pass_2-2.pdf", "pass_2-3.pdf"
]


# #수식
# st.latex("E = mc ^ 2 ")

# #체크 박스
# agree = st.checkbox("동의")
# if agree is True:
#     st.write("동의하셨습니다.")

# #슬라이더
# volume = st.slider("음악 볼륨", 0, 100, 50) #min_value, max_value 필요, 초기 기준 값
# st.write("음악 볼륨은 " + str(volume) + "입니다.")

# #라디오버튼, 셀렉트박스
# #라디오 버튼: 하나만 선택할 수 있음.
# #셀렉트 박스: 여러개 선택 가능

# #라디오 버튼 -> 기본값 항상 존재
# gender = st.radio("성별을 선택하세요.", ["남성", "여성", "밝힐 수 없음"])
# st.write("성별은 " + gender + " 입니다.")

# #셀렉트 박스 -> 기본값이 없음
# flower = st.selectbox("좋아하는 꽃을 모두 선택하세요.", ["---선택---", "장미", "튤립", "국화", "해바라기", "기타"])


# #데이터 표시 및 시각화
# #DataFrame과 표
# import pandas as pd

# df = pd.DataFrame({
#     "학번": ["20170321", "20180111", "20163155", "20190220"],
#     "이름": ["김철수", "최영희", "신지수", "이철민"]
# })

# st.dataframe(df)

# #빈 공간 생성
# st.empty()

# #빈 칸 생성
# st.container(height=100)

# #빈 칸 생성(테두리 없음)
# st.container(border=False, height=100)

# #DataFrame을 테이블로 만들기
# st.table(df)

# #차트 그리기
# import numpy as np

# #랜덤
# # chart_data = pd.DataFrame(
# #     np.random.randn(20, 3),
# #     columns=["a", "b", "c"] 
# #     ) 

# # st.line_chart(chart_data)

# #랜덤 아닌 변수
# chart_data = pd.DataFrame({
#     "국어": [100, 95, 80],
#     "영어": [80, 95, 100],
#     "수학": [95, 100, 80]
# })
# st.line_chart(chart_data)

# #
# import streamlit as st
# import numpy as np

# st.title("간단한 숫자 데이터 분석하기")

# # 사용자로부터 숫자 입력받기
# numbers = st.text_input("숫자 리스트를 입력하세요 (쉼표로 구분)", "1,2,3,4,5")  # 플레이스홀더, 기본값
# number_list = [float(x) for x in numbers.split(",")]

# # 통계 정보 계산
# mean_value = np.mean(number_list)
# median_value = np.median(number_list)
# stdev_value = np.std(number_list)

# # 결과 출력
# st.write(f"평균값: {mean_value}")
# st.write(f"중앙값: {median_value}")
# st.write(f"표준편차: {stdev_value}")