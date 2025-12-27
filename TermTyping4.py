## 0. Setup

#!pip install transformers bitsandbytes accelerate datasets outlines scikit-learn sentence-transformers peft trl

import os
from tqdm import tqdm
import json
import re
from collections import defaultdict
import random
import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
)
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics.pairwise import cosine_similarity
import gc
from peft import PeftModel, PeftConfig
from trl import SFTTrainer, SFTConfig
import trl
import sys
import time
import gc

# --- Force flush output to see logs immediately ---
sys.stdout.reconfigure(line_buffering=True)

print(f"--- DEBUG TRL ---")
print(f"Version chargée : {trl.__version__}")
print(f"Emplacement : {trl.__file__}")
print(f"Python Executable : {sys.executable}")
print(f"-----------------")


root_path = "/home/infres/pprin-23/LLM/TermTyping"

LLM_MODEL = "Google-Large"

"""## 1. Load WordNet Data"""

def WN_TaskA_TextClf_dataset_builder(train_test: str):
    if train_test == "train":
        json_file = root_path + "/WordNet/A.1(FS)_WordNet_Train.json"
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

    elif train_test == "test":
        data_file = root_path + "/WordNet/A.1(FS)_WordNet_Test.json"
        gt_file = root_path + "/WordNet/A.1(FS)_WordNet_Test_GT.json"

        with open(data_file, 'r', encoding='utf-8') as f_data:
            test_data = json.load(f_data)
        with open(gt_file, 'r', encoding='utf-8') as f_gt:
            gt_data = json.load(f_gt)

        gt_lookup = {}
        for item in gt_data:
            label = item['type']
            if isinstance(label, list):
                label = label[0]
            gt_lookup[item['ID']] = label

        data = []
        for item in test_data:
            if item['ID'] in gt_lookup:
                new_item = item.copy()
                new_item['type'] = gt_lookup[item['ID']]
                data.append(new_item)

    types = set()
    for item in data:
        types.add(item["type"])
    labels = list(types)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    text = []
    sentences = []
    label = []
    for item in data:
        sentences.append(str(item["sentence"]))
        text.append(str(item["term"]))
        label.append(label2id[item["type"]])
    print("WordNet")
    print("The total number of data for", train_test, "is: ", len(text))
    print("The total number of labels is: ", len(id2label))
    return id2label, label2id, text, label, sentences

id2label,label2id,text,label,sentences = WN_TaskA_TextClf_dataset_builder("test")

available_labels = list(id2label.values())

device = "cuda" if torch.cuda.is_available() else "cpu"
n_gpus = torch.cuda.device_count()
print(f"Utilisation de {n_gpus} GPUs !")

if LLM_MODEL == "Qwen":
    llm_name_qwen = "Qwen/Qwen3-4B-Instruct-2507"
    
    tokenizer_qwen = AutoTokenizer.from_pretrained(llm_name_qwen, padding_side="left")
    tokenizer_qwen.use_default_system_prompt = False
    tokenizer_qwen.pad_token_id = tokenizer_qwen.eos_token_id

    llm_qwen = AutoModelForCausalLM.from_pretrained(
        llm_name_qwen,
        torch_dtype=torch.float16
    ).to(device)

    if n_gpus > 1:
        llm_qwen = torch.nn.DataParallel(llm_qwen)

    llm_qwen.eval()
    
    generation_config_qwen = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer_qwen.eos_token_id,
        pad_token_id=tokenizer_qwen.pad_token_id,
    )

    def generate_qwen_batched(prompts, llm_model, tokenizer_model, generation_cfg):
        turns_batch = [[{"role": "user", "content": p}] for p in prompts]
        text_batch = [
            tokenizer_model.apply_chat_template(
                turn, 
                tokenize=False,
                add_generation_prompt=True
            ) 
            for turn in turns_batch
        ]
        
        batch_inputs = tokenizer_model(
            text_batch, 
            padding=True, 
            return_tensors="pt"
        ).to(device)

        # FIX: Check if 'module' exists (DataParallel) or use model directly (PeftModel/Single GPU)
        if hasattr(llm_model, "module"):
            model_to_run = llm_model.module
        else:
            model_to_run = llm_model

        generated_ids = model_to_run.generate(
            **batch_inputs, 
            max_new_tokens=generation_cfg.max_new_tokens,
            pad_token_id=tokenizer_model.pad_token_id
        )

        outputs = []
        input_len = batch_inputs["input_ids"].shape[-1]
        for i in range(len(generated_ids)):
            output_ids = generated_ids[i][input_len:]
            content = tokenizer_model.decode(output_ids, skip_special_tokens=True)
            outputs.append(content.strip())
            
        return outputs

elif LLM_MODEL == "Google-Large":
    llm_name_google = "google/flan-t5-large" 
    tokenizer_google = AutoTokenizer.from_pretrained(llm_name_google)
    tokenizer_google.use_default_system_prompt = False
    
    llm_google = AutoModelForSeq2SeqLM.from_pretrained(
        llm_name_google,
        torch_dtype=torch.float16
    ).to(device)

    if n_gpus > 1:
        llm_google = torch.nn.DataParallel(llm_google)
        
    llm_google.eval()

    generation_config_google = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer_google.eos_token_id,
        pad_token_id=tokenizer_google.pad_token_id,
    )

    def generate_google_batched(prompts, llm_model, tokenizer_model, generation_cfg):
        inputs = tokenizer_model(prompts, return_tensors="pt", padding=True).to(device)
        
        # FIX: Check if 'module' attribute actually exists (DataParallel) before accessing it
        # This handles both DataParallel (Training) and standard PeftModel (Evaluation)
        if hasattr(llm_model, "module"):
            model_to_run = llm_model.module
        else:
            model_to_run = llm_model

        generated_ids = model_to_run.generate(
            **inputs, 
            max_new_tokens=generation_cfg.max_new_tokens
        )
        outputs = tokenizer_model.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs
elif LLM_MODEL == "Google-Small":
    print("--- Chargement de Google T5-Small (Mode Compatible Colab) ---")
    llm_name_google = "google/flan-t5-small"
    tokenizer_google = AutoTokenizer.from_pretrained(llm_name_google, padding_side="left")
    
    tokenizer_google.use_default_system_prompt = False
    tokenizer_google.pad_token_id = tokenizer_google.eos_token_id
    
    llm_google = AutoModelForSeq2SeqLM.from_pretrained(
        llm_name_google,
        torch_dtype=torch.float32 
    ).to(device)
    
    llm_google.eval()

    generation_config_google = GenerationConfig(
        max_new_tokens=128, 
        do_sample=False,
    )

    def generate_google_simple(prompt, llm_model, tokenizer_model, generation_cfg):
        inputs = tokenizer_model(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # CORRECTION : input_ids doit être passé en argument nommé
            generated_ids = llm_model.generate(
                input_ids=inputs.input_ids,
                max_new_tokens=generation_cfg.max_new_tokens
            )
        content = tokenizer_model.decode(generated_ids[0], skip_special_tokens=True)
        return content.strip()
elif LLM_MODEL == "Google-Base":
    llm_name_google = "google/flan-t5-base"
    tokenizer_google = AutoTokenizer.from_pretrained(llm_name_google)
    tokenizer_google.use_default_system_prompt = False

    llm_google = AutoModelForSeq2SeqLM.from_pretrained(
        llm_name_google,
        torch_dtype=torch.float16
    ).to(device)

    if n_gpus > 1:
        llm_google = torch.nn.DataParallel(llm_google)

    llm_google.eval()

    generation_config_google = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer_google.eos_token_id,
        pad_token_id=tokenizer_google.pad_token_id,
    )

    # À remplacer dans la section elif LLM_MODEL == "Google-Base":
    def generate_google_batched(prompts, llm_model, tokenizer_model, generation_cfg):
        inputs = tokenizer_model(prompts, return_tensors="pt", padding=True).to(device)
        
        # CORRECTION : Vérifier si l'attribut 'module' existe vraiment
        if hasattr(llm_model, "module"):
            model_to_run = llm_model.module
        else:
            model_to_run = llm_model

        generated_ids = model_to_run.generate(
            **inputs,
            max_new_tokens=generation_cfg.max_new_tokens
        )
        outputs = tokenizer_model.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs

"""## Classification tasks"""

def classify_term_type_with_llm(term, sentence, labels, llm_model, tokenizer_model, few_shot_prompting, active_generate_func, generation_cfg):
    if sentence != "":
      if few_shot_prompting:
        prompt = f"""
        Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose the term type from: {', '.join(labels)}.\n
        Here are some examples to help you:\n
        - term: green, sentence: the car is green, answer: adjective\n
        - term: building, sentence: the building is on fire, answer: noun\n
        - term: walk, sentence: I walk on the street, answer: verb\n
        - term: slowly, sentence: the rat is moving slowly, answer: adverb\n

        Answer by only giving the term type and not explaining.
        """
      else:
        prompt = f"Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose the term type from: {', '.join(labels)}. Answer by only giving the term type and not explaining."
    else:
      if few_shot_prompting:
        prompt = f"""
        What is the type of the term: '{term}'? Choose from: {', '.join(labels)}.\n
        Here are some examples to help you:\n
        - term: green, answer: adjective\n
        - term: building, answer: noun\n
        - term: walk, answer: verb\n
        - term: slowly, answer: adverb\n

        Answer by only giving the term type and not explaining.
        """
      else:
        prompt = f"What is the type of the term: '{term}'? Choose from: {', '.join(labels)}. Answer by only giving the term type and not explaining."

    with torch.no_grad():
      generated_text = active_generate_func(prompt, llm_model, tokenizer_model, generation_cfg)

    generated_text_lower = generated_text.lower()
    for label_name in labels:
        if label_name in generated_text_lower:
            return label_name

    first_word_generated = generated_text_lower.split(" ")[0]
    if first_word_generated in labels:
        return first_word_generated

    return "Unknown or no clear label generated"


def run_classification(classification_task, k):
    if LLM_MODEL == "Google-Large":
        active_llm_model = llm_google
        active_tokenizer_model = tokenizer_google
        active_generation_config = generation_config_google
        active_generate_func = generate_google_batched 
        print("LLM used: Google Flan T5 Large")
    elif LLM_MODEL == "Google-Small":
        active_llm_model = llm_google
        active_tokenizer_model = tokenizer_google
        active_generation_config = generation_config_google
        active_generate_func = generate_google_simple 
        print("LLM used: Google Flan T5 Small")
    elif LLM_MODEL == "Qwen":
        active_llm_model = llm_qwen
        active_tokenizer_model = tokenizer_qwen
        active_generation_config = generation_config_qwen
        active_generate_func = generate_qwen_batched
        print("LLM used: Qwen3 4B Instruct")
    elif LLM_MODEL == "Google-Base":
        active_llm_model = llm_google
        active_tokenizer_model = tokenizer_google
        active_generation_config = generation_config_google
        active_generate_func = generate_google_batched
        print("LLM used: Google Flan T5 Base")

    print(f"Running predictions on Test Set for the classification task {classification_task}")

    if classification_task == "classify_term_type_with_rag":
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loading Wikitext-103...")
        wiki_data = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        wikidata_sentences = [text for text in wiki_data['text'] if text.strip() and len(text.split()) > 5]
        print(f"RAG Knowledge Base loaded with {len(wikidata_sentences)} Wikipedia sentences.")
        print("Encoding Wikidata sentences...")
        wikidata_embeddings = embedder.encode(wikidata_sentences, convert_to_tensor=True).cpu().numpy()
        print("Encoding complete.")
    elif classification_task == "classify_term_type_with_dynamic_few_shot":
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        train_id2label, train_label2id, train_terms, train_labels, train_sentences = WN_TaskA_TextClf_dataset_builder("train")
        print("Encoding Train sentences for Dynamic Few-Shot...")
        train_embeddings = embedder.encode(train_sentences, convert_to_tensor=True).cpu().numpy()
        print("Encoding complete.")

    correct_predictions = 0
    total_predictions = 0
    
    if LLM_MODEL == "Google-Small":
        print("Mode: Single Item Loop (Colab Compatible)")
        pbar = tqdm(zip(text, sentences, label), total=len(text))
        
        for test_term, test_sentence, actual_label_idx in pbar:
            actual_label = id2label[actual_label_idx]
            
            if classification_task == "classify_term_type_with_dynamic_few_shot":
                 predicted_label = classify_term_type_with_dynamic_few_shot(
                    test_term, test_sentence, available_labels, 
                    active_llm_model, active_tokenizer_model, 
                    active_generate_func, active_generation_config,
                    embedder, train_embeddings, train_terms, train_labels, train_sentences, train_id2label, k
                )
            
            if predicted_label == actual_label:
                correct_predictions += 1
            total_predictions += 1

            current_accuracy = correct_predictions / total_predictions
            pbar.set_postfix({f"Accuracy k={k}": f"{current_accuracy * 100:.2f}%"})

    else:
        print("Mode: Batched Execution")
        if LLM_MODEL == "Qwen":
            batch_generate_func = generate_qwen_batched
        else:
            batch_generate_func = generate_google_batched
            
        BATCH_SIZE = 16
        all_indices = list(range(len(text)))
        pbar = tqdm(total=len(text))

        for i in range(0, len(text), BATCH_SIZE):
            batch_indices = all_indices[i : i + BATCH_SIZE]
            batch_terms = [text[j] for j in batch_indices]
            batch_sentences = [sentences[j] for j in batch_indices]
            batch_labels_idx = [label[j] for j in batch_indices]

            if classification_task == "classify_term_type_with_dynamic_few_shot":
                 batch_dynamic_examples = get_dynamic_few_shot_examples_batched(
                    batch_sentences, batch_terms, 
                    embedder, train_embeddings, train_terms, train_labels, train_sentences, 
                    train_id2label, k
                )
            else:
                 batch_dynamic_examples = [""] * len(batch_terms)

            prompts = []
            for idx, (term, sentence) in enumerate(zip(batch_terms, batch_sentences)):
                dynamic_examples = batch_dynamic_examples[idx]
                prompt = f"""Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose from: {', '.join(available_labels)}.\n{dynamic_examples}\nAnswer by only giving the term type."""
                prompts.append(prompt)
            
            batch_responses = batch_generate_func(prompts, active_llm_model, active_tokenizer_model, active_generation_config)
            
            for response, actual_idx in zip(batch_responses, batch_labels_idx):
                resp_lower = response.lower()
                pred = "unknown"
                for lbl in available_labels:
                    if lbl in resp_lower:
                        pred = lbl
                        break
                if pred == id2label[actual_idx]:
                    correct_predictions += 1
                total_predictions += 1
            
            pbar.update(len(batch_indices))
            current_acc = correct_predictions / total_predictions
            pbar.set_postfix({"Accuracy": f"{current_acc * 100:.2f}%"})

    accuracy = correct_predictions / total_predictions
    print(f"\nFinal Accuracy for the classification task {classification_task} and k={k}: {accuracy * 100:.2f}%")

"""## RAG"""

def get_rag_context(query_sentence, embedder, wikidata_embeddings, wikidata_sentences, k):
    if not query_sentence:
        return ""
    query_embedding = embedder.encode([query_sentence], convert_to_tensor=True).cpu().numpy()
    similarities = cosine_similarity(query_embedding, wikidata_embeddings)
    top_k_indices = similarities[0].argsort()[-k:][::-1]
    retrieved_contexts = [wikidata_sentences[idx] for idx in top_k_indices]
    return " ".join(retrieved_contexts)

def classify_term_type_with_rag(term, sentence, labels, llm_model, tokenizer_model, few_shot_prompting, active_generate_func, generation_cfg, embedder, wikidata_embeddings, wikidata_sentences, k):
    retrieved_context = get_rag_context(sentence, embedder, wikidata_embeddings, wikidata_sentences,k)

    context_block = ""
    if retrieved_context:
        context_block = f"Here are other similar sentences to help you: {retrieved_context}\n"

    if sentence != "":
      if few_shot_prompting:
        prompt = f"""
        Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose the term type from: {', '.join(labels)}.\n
        {context_block}
        Here are some examples to help you:\n
        - term: green, sentence: the car is green, answer: adjective\n
        - term: building, sentence: the building is on fire, answer: noun\n
        - term: walk, sentence: I walk on the street, answer: verb\n
        - term: slowly, sentence: the rat is moving slowly, answer: adverb\n

        Answer by only giving the term type and not explaining.
        """
      else:
        prompt = f"{context_block}Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose the term type from: {', '.join(labels)}. Answer by only giving the term type and not explaining."
    else:
      if few_shot_prompting:
        prompt = f"""
        What is the type of the term: '{term}'? Choose from: {', '.join(labels)}.\n
        {context_block}
        Here are some examples to help you:\n
        - term: green, answer: adjective\n
        - term: building, answer: noun\n
        - term: walk, answer: verb\n
        - term: slowly, answer: adverb\n

        Answer by only giving the term type and not explaining.
        """
      else:
        prompt = f"What is the type of the term: '{term}'? Choose from: {', '.join(labels)}. {context_block}Answer by only giving the term type and not explaining."

    with torch.no_grad():
      generated_text = active_generate_func(prompt, llm_model, tokenizer_model, generation_cfg)

    generated_text_lower = generated_text.lower()
    for label_name in labels:
        if label_name in generated_text_lower:
            return label_name

    first_word_generated = generated_text_lower.split(" ")[0]
    if first_word_generated in labels:
        return first_word_generated

    return "Unknown or no clear label generated"

"""## RAG on the Train set"""

def get_dynamic_few_shot_examples(query_sentence, query_term, embedder, train_embeddings, train_terms, train_labels, train_sentences, id2label, k):
    search_query = query_sentence if len(str(query_sentence)) > 3 else query_term

    if not search_query:
        return ""

    query_embedding = embedder.encode([str(search_query)], convert_to_tensor=True).cpu().numpy()
    similarities = cosine_similarity(query_embedding, train_embeddings)
    top_k_indices = similarities[0].argsort()[-k:][::-1]

    examples_text = ""
    for idx in top_k_indices:
        ex_term = train_terms[idx]
        ex_sent = train_sentences[idx]
        raw_label = train_labels[idx]
        ex_label = id2label[raw_label] if isinstance(raw_label, int) else raw_label

        if len(str(ex_sent)) > 3:
            examples_text += f"- term: {ex_term}, sentence: {ex_sent}, answer: {ex_label}\n"

    if not examples_text:
         examples_text = "- term: apple, sentence: I ate an apple, answer: noun\n"

    return examples_text

def get_dynamic_few_shot_examples_batched(batch_sentences, batch_terms, embedder, train_embeddings, train_terms, train_labels, train_sentences, id2label, k):
    queries = []
    for s, t in zip(batch_sentences, batch_terms):
        queries.append(str(s) if len(str(s)) > 3 else str(t))

    if not queries:
        return [""] * len(batch_sentences)

    query_embeddings = embedder.encode(queries, convert_to_tensor=True).cpu().numpy()
    similarities = cosine_similarity(query_embeddings, train_embeddings)
    batch_examples_text = []
    
    for i in range(len(queries)):
        top_k_indices = similarities[i].argsort()[-k:][::-1]
        
        examples_text = ""
        for idx in top_k_indices:
            ex_term = train_terms[idx]
            ex_sent = train_sentences[idx]
            raw_label = train_labels[idx]
            ex_label = id2label[raw_label] if isinstance(raw_label, int) else raw_label
            
            if len(str(ex_sent)) > 3:
                examples_text += f"- term: {ex_term}, sentence: {ex_sent}, answer: {ex_label}\n"
        
        if not examples_text:
             examples_text = "- term: apple, sentence: I ate an apple, answer: noun\n"
        
        batch_examples_text.append(examples_text)

    return batch_examples_text

def classify_term_type_with_dynamic_few_shot(term, sentence, labels, llm_model, tokenizer_model, active_generate_func, generation_cfg, embedder, train_embeddings, train_terms, train_labels, train_sentences, id2label, k):
    dynamic_examples = get_dynamic_few_shot_examples(sentence, term, embedder, train_embeddings, train_terms, train_labels, train_sentences, id2label, k)

    prompt = f"""
Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose the term type from: {', '.join(labels)}.

Here are some examples of similar cases to help you:
{dynamic_examples}

Answer by only giving the term type and not explaining.
"""
    generated_text = active_generate_func(prompt, llm_model, tokenizer_model, generation_cfg)

    generated_text_lower = generated_text.lower()
    for label_name in labels:
        if label_name in generated_text_lower:
            return label_name

    first_word = generated_text_lower.split()[0].strip(".,!:")
    if first_word in labels:
        return first_word

    return "unknown"

"""## Fine tuning"""

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

def format_dataset_for_training(terms, sentences, labels_ids, id2label, tokenizer, model_type):
    available_labels = list(id2label.values())
    
    # Liste pour stocker les données
    dataset_data = []

    for i in range(len(terms)):
        term = terms[i]
        sentence = sentences[i]
        label_text = id2label[labels_ids[i]]

        # Création du Prompt (Input)
        if sentence and len(str(sentence)) > 3:
            prompt = f"Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose from: {', '.join(available_labels)}. Answer by only giving the term type."
        else:
            prompt = f"What is the type of the term: '{term}'? Choose from: {', '.join(available_labels)}. Answer by only giving the term type."

        if model_type == "seq2seq": # POUR T5
            # On tokenise directement ici pour T5
            # L'input est le prompt, la cible (label) est le label_text
            model_input = tokenizer(prompt, max_length=256, truncation=True)
            labels = tokenizer(label_text, max_length=32, truncation=True)
            
            dataset_data.append({
                "input_ids": model_input["input_ids"],
                "attention_mask": model_input["attention_mask"],
                "labels": labels["input_ids"] # T5 calcule la perte par rapport à ça
            })
            
        else: # POUR QWEN (Causal)
            # On garde le format texte simple, le SFTTrainer s'occupe du reste
            text = f"User: {prompt}\nAssistant: {label_text}<|endoftext|>"
            dataset_data.append({"text": text})

    return Dataset.from_list(dataset_data)

def run_finetuning(model_name_key, output_dir="./finetuned_model"):
    print(f"--- Démarrage du Fine-Tuning pour : {model_name_key} ---")
    
    # Récupération des données Train
    train_id2label, train_label2id, train_terms, train_labels, train_sentences = WN_TaskA_TextClf_dataset_builder("train")
    
    # Configuration du modèle
    bnb_config = None
    is_seq2seq = False
    
    if model_name_key == "Google-Small":
        model_id = "google/flan-t5-small"
        is_seq2seq = True
        target_modules = ["q", "v"] 
    elif model_name_key == "Google-Base":
        model_id = "google/flan-t5-base"
        is_seq2seq = True
        target_modules = ["q", "v"]
    elif model_name_key == "Google-Large":
        model_id = "google/flan-t5-large"
        is_seq2seq = True
        target_modules = ["q", "v"]
        # FIX: Disable quantization (None). 
        # The model is small enough to run in full precision/float16 on P100.
        bnb_config = None
    elif model_name_key == "Qwen":
        model_id = "Qwen/Qwen3-4B-Instruct-2507"
        is_seq2seq = False
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_use_double_quant=False,
        )
    else:
        raise ValueError("Modèle inconnu.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Chargement du modèle
    if is_seq2seq:
        model_type_str = "seq2seq"
        # T5
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, # Supporte 8bit si activé pour Large
            device_map="auto",
            torch_dtype=torch.float32 if not bnb_config else torch.float16
        )
    else:
        model_type_str = "causal"
        # Qwen
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            torch_dtype=torch.float16 
        )
        model.config.use_cache = False

    # Préparation du Dataset
    train_dataset = format_dataset_for_training(
        train_terms, train_sentences, train_labels, train_id2label, tokenizer, model_type_str
    )

    # Configuration LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM" if is_seq2seq else "CAUSAL_LM"
    )
    
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- BRANCHE : SÉLECTION DU TRAINER ---
    
    if is_seq2seq:
        # >>> STRATÉGIE T5 (Seq2SeqTrainer) <<<
        print(">>> Utilisation de Seq2SeqTrainer pour T5")
        
        # Data Collator spécial qui gère le padding des inputs ET des labels
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{output_dir}_checkpoints",
            per_device_train_batch_size=8, # T5-Base tient largement en batch 8
            gradient_accumulation_steps=2,
            learning_rate=3e-4, # T5 aime les LR un peu plus élevés
            num_train_epochs=3, # T5 a souvent besoin de plus d'époques que Qwen
            logging_steps=10,
            save_strategy="epoch",
            predict_with_generate=False,
            fp16=False, 
            bf16=False,
            report_to="none",
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
    else:
        # >>> STRATÉGIE QWEN (SFTTrainer / Causal) <<<
        print(">>> Utilisation de SFTTrainer pour Qwen")
        
        training_args = SFTConfig(
            output_dir=f"{output_dir}_checkpoints",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            num_train_epochs=1,
            save_strategy="epoch",
            fp16=False,
            bf16=False,
            dataset_text_field="text",
            max_seq_length=256,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            processing_class=tokenizer 
        )

    # Lancement
    trainer.train()
    
    print(f"Sauvegarde de l'adaptateur dans {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Nettoyage
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_dir

from transformers import BitsAndBytesConfig # Assurez-vous d'avoir cet import

def evaluate_finetuned_model(model_name_key, adapter_path, use_rag=False, k=3, batch_size=16):
    print(f"--- Évaluation : {model_name_key} (RAG={use_rag}, k={k}, Batch Size={batch_size}) ---")
    
    # 1. Configuration du Modèle
    if model_name_key == "Google-Small":
        base_model_id = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float32).to("cuda")
        gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)
        gen_func = generate_google_simple 
        
    elif model_name_key == "Google-Base":
        base_model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
        gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)
        gen_func = generate_google_batched

    elif model_name_key == "Qwen":
        base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
        
        # padding_side="left" est CRITIQUE pour la génération batched decoder-only
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # --- FIX OOM : Chargement 4-bit pour l'Inférence ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            quantization_config=bnb_config, # <--- C'est ça qui sauve la VRAM
            device_map="auto"
        )
        gen_config = GenerationConfig(max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        gen_func = generate_qwen_batched
        
    elif model_name_key == "Google-Large":
        base_model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        gen_config = GenerationConfig(max_new_tokens=64)
        gen_func = generate_google_batched

    print(f"Chargement des poids LoRA depuis {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 2. Chargement du RAG (Embeddings)
    embedder = None
    train_embeddings = None
    train_terms = []
    train_labels = []
    train_sentences = []
    
    if use_rag:
        print("Chargement du Train Set et encodage pour le RAG...")
        # --- FIX OOM : Embedder sur CPU ---
        # On force l'embedder sur CPU pour laisser toute la VRAM à Qwen
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        
        t_id2label, t_label2id, train_terms, train_labels, train_sentences = WN_TaskA_TextClf_dataset_builder("train")
        
        # Encodage sur CPU (plus lent mais ne plante pas)
        train_embeddings = embedder.encode(train_sentences, convert_to_tensor=True).cpu().numpy()
        print("Index RAG prêt.")

    # 3. Boucle de Prédiction par Batch
    available_labels = list(id2label.values())
    correct_predictions = 0
    total_predictions = 0
    
    is_batched_func = "batched" in gen_func.__name__
    all_indices = list(range(len(text)))
    pbar = tqdm(total=len(text))

    for i in range(0, len(text), batch_size):
        # A. Préparation du Batch
        batch_indices = all_indices[i : i + batch_size]
        batch_terms = [text[j] for j in batch_indices]
        batch_sentences = [sentences[j] for j in batch_indices]
        batch_labels_idx = [label[j] for j in batch_indices]

        # B. Récupération RAG (Batched)
        batch_dynamic_examples = [""] * len(batch_terms)
        if use_rag:
            batch_dynamic_examples = get_dynamic_few_shot_examples_batched(
                batch_sentences, batch_terms, 
                embedder, train_embeddings, train_terms, train_labels, train_sentences, 
                id2label, k
            )

        # C. Construction des Prompts
        prompts = []
        for idx, (term, sentence) in enumerate(zip(batch_terms, batch_sentences)):
            dynamic_examples = batch_dynamic_examples[idx]
            if dynamic_examples:
                dynamic_examples = f"\nHere are some similar examples to help you:\n{dynamic_examples}\n"
            
            if sentence and len(str(sentence)) > 3:
                base_prompt = f"Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose from: {', '.join(available_labels)}."
            else:
                base_prompt = f"What is the type of the term: '{term}'? Choose from: {', '.join(available_labels)}."

            final_prompt = f"{base_prompt}{dynamic_examples} Answer by only giving the term type."
            prompts.append(final_prompt)

        # D. Génération (Batched)
        with torch.no_grad():
            if is_batched_func:
                batch_responses = gen_func(prompts, model, tokenizer, gen_config)
            else:
                batch_responses = [gen_func(p, model, tokenizer, gen_config) for p in prompts]

        # E. Vérification
        for response, actual_idx in zip(batch_responses, batch_labels_idx):
            resp_lower = response.lower()
            pred = "unknown"
            for lbl in available_labels:
                if lbl in resp_lower:
                    pred = lbl
                    break
            
            if pred == "unknown":
                first_word = resp_lower.split()[0].strip(".,!:")
                if first_word in available_labels:
                    pred = first_word

            if pred == id2label[actual_idx]:
                correct_predictions += 1
            total_predictions += 1
        
        pbar.update(len(batch_indices))
        pbar.set_postfix({"Acc": f"{correct_predictions/total_predictions*100:.2f}%"})

    final_acc = correct_predictions/total_predictions*100
    print(f"Final Accuracy ({model_name_key} + FT + RAG={use_rag}): {final_acc:.2f}%")
    
    # --- NETTOYAGE MÉMOIRE (CRUCIAL) ---
    del model
    del base_model
    if use_rag:
        del embedder
    torch.cuda.empty_cache()
    gc.collect()

import time
import gc
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

def benchmark_all_methods(adapter_path, n_samples=10):
    print(f"\n=== BENCHMARK DE VITESSE (Moyenne sur {n_samples} échantillons) ===")
    
    # 0. Sélection des échantillons
    sample_indices = range(min(n_samples, len(text)))
    sample_terms = [text[i] for i in sample_indices]
    sample_sentences = [sentences[i] for i in sample_indices]
    
    # 1. Chargement du Modèle de BASE
    print(f"--> Chargement du modèle de base : {LLM_MODEL}...")
    
    if LLM_MODEL == "Qwen":
        base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 4-bit obligatoire pour éviter OOM si RAG actif
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
        gen_config = GenerationConfig(max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        gen_func_batched = generate_qwen_batched

    elif LLM_MODEL == "Google-Base":
        base_model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
        gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)
        gen_func_batched = generate_google_batched
        
    elif LLM_MODEL == "Google-Large":
        base_model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
        gen_config = GenerationConfig(max_new_tokens=64)
        gen_func_batched = generate_google_batched
        
    else: # Small
        base_model_id = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float32).to("cuda")
        gen_config = GenerationConfig(max_new_tokens=64)
        # Small n'a pas de fonction batched native dans ce script
        def generate_google_fake_batched(prompts, m, t, c):
            return [generate_google_simple(p, m, t, c) for p in prompts]
        gen_func_batched = generate_google_fake_batched

    # --- CRÉATION DU WRAPPER (CORRECTION DU BUG 'LIST') ---
    # Transforme une fonction qui prend/rend une liste en fonction qui prend/rend un string
    # pour être compatible avec classify_term_type_with_...
    def gen_func_single(prompt, m, t, c):
        # On met le prompt dans une liste, et on récupère le 1er élément de la réponse
        return gen_func_batched([prompt], m, t, c)[0]

    results = {}

    # --- TEST 1 : CLASSIC LLM ---
    print("\n[1/5] Test: Classic LLM (Zero-Shot)...")
    torch.cuda.empty_cache()
    start = time.time()
    for t, s in zip(sample_terms, sample_sentences):
        classify_term_type_with_llm(
            t, s, available_labels, model, tokenizer, False, 
            gen_func_single, # <--- On utilise le wrapper ici
            gen_config
        )
    duration = time.time() - start
    results["Classic LLM"] = duration / n_samples
    print(f"   -> Temps moyen : {results['Classic LLM']:.4f} s/sample")

    # --- PRÉPARATION RAG (CPU) ---
    print("\n[Chargement des données RAG sur CPU...]")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    
    wiki_data = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    wiki_sentences = [txt for txt in wiki_data['text'] if txt.strip() and len(txt.split()) > 5]
    # On limite à 5000 phrases pour que l'encodage du benchmark soit rapide
    wiki_emb = embedder.encode(wiki_sentences[:5000], convert_to_tensor=True).cpu().numpy()
    
    t_id2label, t_label2id, tr_terms, tr_labels, tr_sentences = WN_TaskA_TextClf_dataset_builder("train")
    train_emb = embedder.encode(tr_sentences, convert_to_tensor=True).cpu().numpy()

    # --- TEST 2 : RAG WIKIPEDIA ---
    print("\n[2/5] Test: RAG Wikipedia...")
    torch.cuda.empty_cache()
    start = time.time()
    for t, s in zip(sample_terms, sample_sentences):
        classify_term_type_with_rag(
            t, s, available_labels, model, tokenizer, False, 
            gen_func_single, # <--- Wrapper
            gen_config,
            embedder, wiki_emb, wiki_sentences[:5000], k=3
        )
    duration = time.time() - start
    results["RAG Wikipedia"] = duration / n_samples
    print(f"   -> Temps moyen : {results['RAG Wikipedia']:.4f} s/sample")
    
    del wiki_emb, wiki_sentences
    gc.collect()

    # --- TEST 3 : RAG TRAIN SET ---
    print("\n[3/5] Test: RAG Train Set...")
    torch.cuda.empty_cache()
    start = time.time()
    for t, s in zip(sample_terms, sample_sentences):
        classify_term_type_with_dynamic_few_shot(
            t, s, available_labels, model, tokenizer, 
            gen_func_single, # <--- Wrapper
            gen_config,
            embedder, train_emb, tr_terms, tr_labels, tr_sentences, t_id2label, k=3
        )
    duration = time.time() - start
    results["RAG Train Set"] = duration / n_samples
    print(f"   -> Temps moyen : {results['RAG Train Set']:.4f} s/sample")

    # --- TEST 4 : FINE-TUNED MODEL ---
    print("\n[4/5] Chargement Adaptateur & Test Fine-Tuned...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    torch.cuda.empty_cache()
    start = time.time()
    # Ici on reconstruit manuellement les prompts pour utiliser le mode batched natif si on veut
    # Ou on continue avec gen_func_single pour être cohérent
    for t, s in zip(sample_terms, sample_sentences):
        classify_term_type_with_llm(
            t, s, available_labels, model, tokenizer, False, 
            gen_func_single, 
            gen_config
        )
    duration = time.time() - start
    results["Fine-Tuned"] = duration / n_samples
    print(f"   -> Temps moyen : {results['Fine-Tuned']:.4f} s/sample")

    # --- TEST 5 : FINE-TUNED + RAG ---
    print("\n[5/5] Test: Fine-Tuned + RAG Train...")
    torch.cuda.empty_cache()
    start = time.time()
    for t, s in zip(sample_terms, sample_sentences):
         classify_term_type_with_dynamic_few_shot(
            t, s, available_labels, model, tokenizer, 
            gen_func_single, 
            gen_config,
            embedder, train_emb, tr_terms, tr_labels, tr_sentences, t_id2label, k=3
        )
    duration = time.time() - start
    results["FT + RAG Train"] = duration / n_samples
    print(f"   -> Temps moyen : {results['FT + RAG Train']:.4f} s/sample")

    # --- RESUME ---
    print("\n" + "="*50)
    print(f"RESULTATS INFERENCE ({LLM_MODEL}, Moyenne sur {n_samples} items)")
    print("="*50)
    for method, timing in results.items():
        print(f"{method:<30} | {timing:.4f} s")
    print("="*50 + "\n")

    del model, embedder, train_emb
    torch.cuda.empty_cache()
    gc.collect()

    return results

from collections import Counter

def perform_error_analysis(adapter_path, n_samples=50):
    print(f"\n=======================================================")
    print(f"   ANALYSE D'ERREURS DÉTAILLÉE (Sur {n_samples} échantillons)")
    print(f"=======================================================")
    
    # 0. Sélection des données
    # On prend un subset pour que l'analyse soit rapide
    indices = range(min(n_samples, len(text)))
    sample_terms = [text[i] for i in indices]
    sample_sentences = [sentences[i] for i in indices]
    sample_labels_ids = [label[i] for i in indices]
    
    # 1. Chargement du Modèle (Config Optimisée comme le Benchmark)
    print(f"--> Chargement du modèle : {LLM_MODEL}...")
    if LLM_MODEL == "Qwen":
        base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
        gen_config = GenerationConfig(max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        gen_func_batched = generate_qwen_batched
    elif LLM_MODEL == "Google-Base":
        base_model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
        gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)
        gen_func_batched = generate_google_batched
    else: 
        # Fallback pour les autres T5
        base_model_id = "google/flan-t5-large" if LLM_MODEL == "Google-Large" else "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        dtype = torch.float16 if LLM_MODEL == "Google-Large" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=dtype, device_map="auto" if dtype==torch.float16 else "cuda")
        gen_config = GenerationConfig(max_new_tokens=64)
        if LLM_MODEL == "Google-Small":
            gen_func_batched = lambda p, m, t, c: [generate_google_simple(x, m, t, c) for x in p]
        else:
            gen_func_batched = generate_google_batched

    # Wrapper pour compatibilité (List -> String)
    def gen_func_single(prompt, m, t, c):
        return gen_func_batched([prompt], m, t, c)[0]

    # 2. Préparation RAG (CPU)
    print("--> Chargement Index RAG (CPU)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    t_id2label, t_label2id, tr_terms, tr_labels, tr_sentences = WN_TaskA_TextClf_dataset_builder("train")
    # On encode tout le train set ou un subset selon besoins de précision
    train_emb = embedder.encode(tr_sentences, convert_to_tensor=True).cpu().numpy()

    # --- FONCTION D'ANALYSE INTERNE ---
    def run_analysis_loop(method_name, prediction_lambda):
        print(f"\n>>> Analyse Méthode : {method_name}")
        errors = []
        correct = 0
        
        # On utilise tqdm pour voir la progression
        for i in tqdm(range(len(sample_terms)), leave=False):
            t, s, l_idx = sample_terms[i], sample_sentences[i], sample_labels_ids[i]
            actual = id2label[l_idx]
            
            # Prédiction
            try:
                pred = prediction_lambda(t, s)
            except Exception as e:
                pred = "ERROR"
            
            # Normalisation pour comparaison
            pred_clean = pred.lower().strip()
            # Nettoyage basique (ex: "noun." -> "noun")
            for lbl in available_labels:
                if lbl in pred_clean:
                    pred_clean = lbl
                    break
            
            if pred_clean == actual:
                correct += 1
            else:
                errors.append({
                    "term": t, "sentence": s, 
                    "expected": actual, "predicted": pred_clean, "raw": pred
                })

        accuracy = correct / len(sample_terms) * 100
        print(f"   Accuracy: {accuracy:.2f}% ({correct}/{len(sample_terms)})")
        
        if errors:
            # Analyse des Confusions
            confusions = Counter([(e['expected'], e['predicted']) for e in errors])
            print("   [Confusions Fréquentes] (Attendu -> Prédit) :")
            for (exp, prd), count in confusions.most_common(3):
                print(f"     - {exp} -> {prd} : {count} fois")
            
            # Exemples d'erreurs
            print("   [Exemples d'Erreurs] :")
            for e in errors[:3]: # Affiche les 3 premières erreurs
                print(f"     * Terme: '{e['term']}'")
                print(f"       Phrase: {e['sentence']}")
                print(f"       ❌ Prédit: {e['predicted']} (Brut: '{e['raw']}') | ✅ Attendu: {e['expected']}")
                print("       ---")
        else:
            print("   🎉 Aucune erreur sur cet échantillon !")
            
        return accuracy

    # --- EXECUTION DES SCENARIOS ---

    # 1. Base Zero-Shot
    run_analysis_loop("Base LLM (Zero-Shot)", 
        lambda t, s: classify_term_type_with_llm(
            t, s, available_labels, model, tokenizer, False, gen_func_single, gen_config
        )
    )

    # 2. Base + RAG (Train Set)
    run_analysis_loop("Base LLM + RAG (Dynamic Few-Shot)", 
        lambda t, s: classify_term_type_with_dynamic_few_shot(
            t, s, available_labels, model, tokenizer, gen_func_single, gen_config,
            embedder, train_emb, tr_terms, tr_labels, tr_sentences, t_id2label, k=3
        )
    )

    # 3. Chargement Fine-Tuned
    print("\n--> Application de l'adaptateur Fine-Tuned...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # 4. Fine-Tuned Zero-Shot
    run_analysis_loop("Fine-Tuned (Zero-Shot)", 
        lambda t, s: classify_term_type_with_llm(
            t, s, available_labels, model, tokenizer, False, gen_func_single, gen_config
        )
    )

    # 5. Fine-Tuned + RAG
    run_analysis_loop("Fine-Tuned + RAG", 
        lambda t, s: classify_term_type_with_dynamic_few_shot(
            t, s, available_labels, model, tokenizer, gen_func_single, gen_config,
            embedder, train_emb, tr_terms, tr_labels, tr_sentences, t_id2label, k=3
        )
    )

    # Nettoyage
    del model, embedder, train_emb
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    #for k in range(1,11):
    #    run_classification("classify_term_type_with_dynamic_few_shot", k)
    #for k in range(1,11):
    #    run_classification("classify_term_type_with_rag", k)
    #adapter_path = run_finetuning("Google-Large", output_dir="./ft_google_large")
    #evaluate_finetuned_model("Google-Large", "./ft_google_large", use_rag=False, k=0)
    #for i in range(1,11):
    #    evaluate_finetuned_model("Qwen", "./ft_qwen", use_rag=True, k=i)
    #run_classification("classify_term_type_with_llm", k=0)

    #path_ft = "./ft_google_small"
    
    # Lancement du benchmark
    #benchmark_all_methods(path_ft, n_samples=10)

    # Remplacez par votre dossier d'adaptateur
    path_ft = "./ft_google_large" 
    
    # Lance l'analyse sur 50 phrases au hasard (rapide)
    perform_error_analysis(path_ft, n_samples=100)