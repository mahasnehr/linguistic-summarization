# Emotion Label Enhancement with Fuzzy Signature Inference System
In this work we propose a data-driven approach based on NN, without relying heavily on human help. We will develop a general framework for single-document summarization. We will train the models on large scale corpora containing thousands of document-summary pairs. 

## Author : Ruba ALMahasneh 

## EMail Address :Mahasnehr@TMIT.BME.HU


## Problem Formulation
Humans express their emotions by words. Emotion analysis is a method to identify emotional status. Emotion Distribution Learning (EDL) translates human emotions by considering multiple intensities conveyed in data. Gen-erally, emotion detection is more complex than just labeling it under six cate-gories: love, fear, joy, sadness, surprise and anger. Indeed, capturing the in-herent uncertainty and vagueness of human emotions is a complex task as presented in Plutchik wheel of emotions. In this paper we propose a novel emotion classification (analysis) framework; the Fuzzy Signature Emotion Classification (FSEC) model. FSEC facilitates Fuzzy Signature (FSig) to dy-namically conduct structural trees depending on the Plutchik wheel of emo-tions, integrated with the classical EDL approach to enhance (to some extent) the classical emotion classification methods. 

## Note:
to run this model Please:


Classical model’s dataset: Each entry in this comprehensive collection features a text segment extracted from Twitter, accompanied by a corresponding label de-noting the predominant emotion conveyed by the message. The emotions are thoughtfully categorized into six distinct classes: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5). The original dataset has over 416809 Twitter messages covering a diverse range of topics and user demographics, each tweet is classified in one of 6 emotions – love, fear, joy, sadness, surprise and anger (full code can be found on GitHub for the two models explained here).


The FSEC model’ dataset: was extended on love emotion (for simplicity but it can be done for all emotions in the wheel) the classes our annotators used are: sadness (0), joy (1), Love (2) is replaced with Lovesub01=serenity, joy, ecsta-sy|Lovesub02= acceptance, trust, and admiration, anger (3), fear (4), and surprise (5).

