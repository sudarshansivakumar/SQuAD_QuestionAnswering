# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:49:21 2021

@author: Sudarshan_backup
"""

import pandas as pd
import numpy as np 
import torch 
import streamlit as st


from transformers import pipeline,QuestionAnsweringPipeline, DistilBertForQuestionAnswering,AutoTokenizer

model_checkpoint = "distilbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model_path = "C:/Users/Sudarshan_backup/Desktop/Datasets/QA_Model/PreTrained_epoch2"
myQAModel = DistilBertForQuestionAnswering.from_pretrained(model_path)

QAPipeline = QuestionAnsweringPipeline(model = myQAModel,tokenizer = tokenizer)

#example_context = "From childhood through most of his professional career, Nadal was coached by his uncle Toni. He was one of the most successful teenagers in ATP Tour history, reaching No. 2 in the world at age 19 and winning 16 titles, including his first French Open and six Masters events. Nadal became No. 1 for the first time in 2008 after his first major victory off clay against the longtime top-ranked Federer, his main rival through 2010, in a historic Wimbledon final. He also won an Olympic gold medal in singles that year in Beijing. After defeating Djokovic in the 2010 US Open final, the 24-year-old Nadal became the youngest male tennis player in the Open Era to achieve the career Grand Slam, and also became the first male tennis player to win three Grand Slams on three different surfaces (hard, grass and clay) the same calendar year. With his Olympic gold medal, he is also one of only two male players to complete the career Golden Slam"
#example_question = "Where did he win an Olympics prize?"

#pipe_outputs = QAPipeline(question = example_question,context = example_context,topk = 3, max_seq_len = 512)

#print(pipe_outputs)
#@st.cache
st.markdown("Hello, I am Sudarshan Sivakumar, a 3rd year Undergraduate in Computer Science and Engineering at the Manipal Institute of Technology. This is a Question Answering System I made using the Hugging Face transformers library on the Stanford Question Answering dataset. The model has been deployed using Streamlit")

st.markdown("If you enter a context passage and ask a question whose answer is in the context, the program will give you an answer span from the passage which best answers the question")


context = st.text_area("Context Paragraph", "")
question = st.text_input("Question", "")

if context:
    # Execute question against paragraph
    if question:
        outputs = QAPipeline(question = question,context = context,topk = 3, max_seq_len = 512)
        answer = outputs[0]["answer"]
        output_answer = st.text_area("Answer",answer)
        