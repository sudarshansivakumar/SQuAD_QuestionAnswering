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

# The model_path here would be the directory in which you saved the model using the HuggingFace model.save_pretrained() function
model_path = "C:/Users/Sudarshan_backup/Desktop/Datasets/QA_Model/PreTrained_epoch2"
myQAModel = DistilBertForQuestionAnswering.from_pretrained(model_path)

QAPipeline = QuestionAnsweringPipeline(model = myQAModel,tokenizer = tokenizer)

# This is a markdown message at the beginning of my application in which I'm introducing myself and explaining the question. You should add whatever message you want to. 
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
        
