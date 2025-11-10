# Intuitionistic Fuzzy Approach for Emotion Analysis Enhancement
In this work we propose a data-driven approach based on NN, without relying heavily on human help. We will develop a general framework for single-document summarization. We will train the models on large scale corpora containing thousands of document-summary pairs. 

## Author : Ruba ALMahasneh 

## EMail Address :Mahasnehr@TMIT.BME.HU


## Problem Formulation
In this study, we introduced a novel intuitionisitic fuzzy-based approach to classify emotions derived from Plutchik wheel of emotions. The Intuitionistic Fuzzy Set Emotion Classification (IFSEC) was illustrated on love as a complex emotion (derived from the combination of six key sub-emotions: serenity, joy, ecstasy, acceptance, trust, and admiration). Using intuitionisitic fuzzy sets, we captured the overlapping nature of these emotions in textual data, allowing smooth transitions between different levels of emotional intensity. Then we extended the model (IFSEC) using Mamdani inference system and aggregated different tree structures (observations) where each instance modeled emotions (love emotion in particular) slightly different on the emotions spectrum. Finally, the model was utilized to generate a labeled dataset and tested on Recurrent Neural Networks model for emotion detection. IFSEC exhibits the ability in capturing the inherent ambiguity and variability in emotional states and improves the accuracy of emotion labeling. We are eager to expand this model for future applications in sentiment analysis and psychological studies where accurately understanding emotional states is vital. 

## Note:
to run this model Please:


Classical model’s dataset: Each entry in this comprehensive collection features a text segment extracted from Twitter, accompanied by a corresponding label de-noting the predominant emotion conveyed by the message. The emotions are thoughtfully categorized into six distinct classes: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5). The original dataset has over 416809 Twitter messages covering a diverse range of topics and user demographics, each tweet is classified in one of 6 emotions – love, fear, joy, sadness, surprise and anger (full code can be found on GitHub for the two models explained here).


The FSEC model’ dataset: was extended on love emotion (for simplicity but it can be done for all emotions in the wheel) the classes our annotators used are: sadness (0), joy (1), Love (2) is replaced with Lovesub01=serenity, joy, ecsta-sy|Lovesub02= acceptance, trust, and admiration, anger (3), fear (4), and surprise (5).

