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

LLM_MODEL = "Qwen"

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

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def calculate_metrics(y_true, y_pred, model_name="Model"):
    print(f"\n--- 📊 RAPPORT DE PERFORMANCE CORRIGÉ : {model_name} ---")
    
    # 1. On définit explicitement les 4 vraies classes (sans 'unknown')
    # Cela force sklearn à ignorer 'unknown' dans le calcul de la moyenne Macro
    true_labels = sorted(list(set([l for l in y_true if l != "unknown"])))
    
    # 2. Rapport détaillé
    # 'labels=true_labels' force l'affichage uniquement des vraies classes
    print(classification_report(y_true, y_pred, labels=true_labels, digits=4, zero_division=0))
    
    # 3. Métriques Macro (Sur 4 classes uniquement)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, 
        average='macro', 
        labels=true_labels, # <--- C'est la clé : on exclut 'unknown' du diviseur
        zero_division=0
    )
    
    print(f"🏆 RÉSUMÉ LEADERBOARD ({model_name}):")
    print(f"  - Accuracy:       {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"  - Macro F1:       {f1*100:.2f}%  <-- SCORE RÉEL (sur 4 classes)")
    print(f"  - Macro Precision:{precision*100:.2f}%")
    print(f"  - Macro Recall:   {recall*100:.2f}%")
    print("-" * 50)

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
    
# --- FONCTIONS DE GÉNÉRATION GLOBALES (Hors des if/elif) ---

def generate_qwen_batched(prompts, llm_model, tokenizer_model, generation_cfg):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Template Chat pour Qwen
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

    # Gestion DataParallel / Peft
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

def generate_google_batched(prompts, llm_model, tokenizer_model, generation_cfg):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inputs = tokenizer_model(prompts, return_tensors="pt", padding=True).to(device)
    
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

def generate_google_simple(prompt, llm_model, tokenizer_model, generation_cfg):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inputs = tokenizer_model(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = llm_model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=generation_cfg.max_new_tokens
        )
    content = tokenizer_model.decode(generated_ids[0], skip_special_tokens=True)
    return content.strip()

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
    print(f"\n==================================================================")
    print(f"   ÉVALUATION BASELINE : {LLM_MODEL} | Tâche : {classification_task}")
    print(f"==================================================================")

    # 1. Sélection Modèle & Générateur
    if LLM_MODEL == "Google-Large":
        active_llm_model = llm_google
        active_tokenizer_model = tokenizer_google
        active_generation_config = generation_config_google
        active_generate_func = generate_google_batched 
    elif LLM_MODEL == "Google-Small":
        active_llm_model = llm_google
        active_tokenizer_model = tokenizer_google
        active_generation_config = generation_config_google
        active_generate_func = generate_google_simple 
    elif LLM_MODEL == "Qwen":
        active_llm_model = llm_qwen
        active_tokenizer_model = tokenizer_qwen
        active_generation_config = generation_config_qwen
        active_generate_func = generate_qwen_batched
    elif LLM_MODEL == "Google-Base":
        active_llm_model = llm_google
        active_tokenizer_model = tokenizer_google
        active_generation_config = generation_config_google
        active_generate_func = generate_google_batched

    # 2. Setup RAG
    embedder = None
    rag_corpus = None
    rag_embeddings = None
    
    if "rag" in classification_task or "few_shot" in classification_task:
        print("--> Chargement RAG (CPU)...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        
        if classification_task == "classify_term_type_with_rag": # WIKIPEDIA
            wiki_data = load_dataset("wikitext", "wikitext-103-v1", split="validation")
            wikidata_sentences = [text for text in wiki_data['text'] if text.strip() and len(text.split()) > 5][:5000]
            rag_embeddings = embedder.encode(wikidata_sentences, convert_to_tensor=True).cpu().numpy()
            rag_corpus = wikidata_sentences
            print("   (Wikidata Index Prêt)")
            
        elif classification_task == "classify_term_type_with_dynamic_few_shot": # TRAIN SET
            t_id2label, t_label2id, tr_terms, tr_labels, tr_sentences = WN_TaskA_TextClf_dataset_builder("train")
            rag_embeddings = embedder.encode(tr_sentences, convert_to_tensor=True).cpu().numpy()
            rag_corpus = {
                "terms": tr_terms, "sentences": tr_sentences, 
                "labels": tr_labels, "id2label": t_id2label
            }
            print("   (TrainSet Index Prêt)")

    # 3. Boucle d'Inférence
    y_true_all = []
    y_pred_all = []
    
    BATCH_SIZE = 16
    all_indices = list(range(len(text)))
    
    print("--> Démarrage de l'inférence...")
    pbar = tqdm(total=len(text))

    for i in range(0, len(text), BATCH_SIZE):
        batch_indices = all_indices[i : i + BATCH_SIZE]
        batch_terms = [text[j] for j in batch_indices]
        batch_sentences = [sentences[j] for j in batch_indices]
        batch_labels_idx = [label[j] for j in batch_indices]

        # Préparation Contextes RAG
        batch_context_text = [""] * len(batch_terms)
        
        if classification_task == "classify_term_type_with_dynamic_few_shot":
             batch_context_text = get_dynamic_few_shot_examples_batched(
                batch_sentences, batch_terms, 
                embedder, rag_embeddings, rag_corpus["terms"], rag_corpus["labels"], rag_corpus["sentences"], 
                rag_corpus["id2label"], k
            )
             
        elif classification_task == "classify_term_type_with_rag":
            # On doit récupérer le contexte pour chaque phrase du batch
            # Note: Vous devrez implémenter une version batched de get_rag_context pour l'efficacité
            current_batch_contexts = []
            for sent in batch_sentences:
                # Appel à votre fonction existante (non-batchée mais fonctionnelle)
                ctx = get_rag_context(sent, embedder, rag_embeddings, rag_corpus, k)
                current_batch_contexts.append(ctx)
            batch_context_text = current_batch_contexts

        # Construction Prompts
        prompts = []
        for idx, (term, sentence) in enumerate(zip(batch_terms, batch_sentences)):
            ctx = batch_context_text[idx]
            if ctx: ctx = f"\nExamples:\n{ctx}\n"
            
            if sentence: prompt = f"Given '{term}' in '{sentence}', type? Options: {', '.join(available_labels)}.{ctx} Answer type only."
            else: prompt = f"Type of '{term}'? Options: {', '.join(available_labels)}.{ctx} Answer type only."
            prompts.append(prompt)

        # Génération
        if "batched" in active_generate_func.__name__:
            batch_responses = active_generate_func(prompts, active_llm_model, active_tokenizer_model, active_generation_config)
        else:
            batch_responses = [active_generate_func(p, active_llm_model, active_tokenizer_model, active_generation_config) for p in prompts]

        # Parsing (CORRIGÉ ICI)
        for response, actual_idx in zip(batch_responses, batch_labels_idx):
            resp_lower = response.lower() if response else "" # Protection contre None
            pred = "unknown"
            
            for lbl in available_labels:
                if lbl in resp_lower:
                    pred = lbl; break
            
            if pred == "unknown":
                # --- FIX CRASH : Vérifier que la liste n'est pas vide ---
                split_resp = resp_lower.split()
                if split_resp: # Si la liste n'est pas vide
                    fw = split_resp[0].strip(".,!:")
                    if fw in available_labels: pred = fw
                # Sinon pred reste "unknown"
            
            y_pred_all.append(pred)
            y_true_all.append(id2label[actual_idx])
            
        pbar.update(len(batch_indices))
    
    pbar.close()
    
    # 4. Métriques
    calculate_metrics(y_true_all, y_pred_all, model_name=f"{LLM_MODEL} - {classification_task}")
    
    # Nettoyage
    del embedder, rag_corpus, rag_embeddings
    gc.collect()

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
        
        # 1. On crée la config SANS max_seq_length pour éviter le crash TypeError
        training_args = SFTConfig(
            output_dir=f"{output_dir}_checkpoints",
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            num_train_epochs=3,
            save_strategy="epoch",
            fp16=False,
            bf16=False,
            dataset_text_field="text",
            group_by_length=True,
            report_to="none",
        )

        # 2. On l'injecte manuellement "de force" (C'est sale mais ça marche à tous les coups)
        training_args.max_seq_length = 256

        # 3. On initialise le Trainer SANS max_seq_length
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,        # Le trainer lira args.max_seq_length ici
            processing_class=tokenizer,
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
    print(f"\n==================================================================")
    print(f"   ÉVALUATION FINE-TUNED : {model_name_key} (RAG={use_rag}, k={k})")
    print(f"==================================================================")
    
    # 1. Configuration du Modèle Base (Optimisé Mémoire)
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
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        # FIX OOM : Qwen en 4-bit OBLIGATOIRE pour l'inférence avec RAG
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False
        )
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
        gen_config = GenerationConfig(max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        gen_func = generate_qwen_batched
    elif model_name_key == "Google-Large":
        base_model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
        gen_config = GenerationConfig(max_new_tokens=64)
        gen_func = generate_google_batched

    # 2. Chargement de l'Adaptateur LoRA
    print(f"--> Chargement des poids LoRA depuis {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    except Exception as e:
        print(f"ERREUR CRITIQUE: Impossible de charger l'adaptateur. Vérifiez le chemin. ({e})")
        return

    # 3. Préparation RAG (CPU pour économiser GPU)
    embedder = None
    train_embeddings = None
    if use_rag:
        print("--> Chargement Index RAG (sur CPU)...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        t_id2label, t_label2id, train_terms, train_labels, train_sentences = WN_TaskA_TextClf_dataset_builder("train")
        train_embeddings = embedder.encode(train_sentences, convert_to_tensor=True).cpu().numpy()

    # 4. Boucle d'Inférence (Batched)
    available_labels = list(id2label.values())
    y_true_all = []
    y_pred_all = []
    
    is_batched_func = "batched" in gen_func.__name__
    all_indices = list(range(len(text))) # text = Test terms globaux
    
    print("--> Démarrage de l'inférence...")
    pbar = tqdm(total=len(text))

    for i in range(0, len(text), batch_size):
        batch_indices = all_indices[i : i + batch_size]
        batch_terms = [text[j] for j in batch_indices]
        batch_sentences = [sentences[j] for j in batch_indices]
        batch_labels_idx = [label[j] for j in batch_indices]

        # A. Contexte RAG
        batch_dynamic_examples = [""] * len(batch_terms)
        if use_rag:
            batch_dynamic_examples = get_dynamic_few_shot_examples_batched(
                batch_sentences, batch_terms, 
                embedder, train_embeddings, train_terms, train_labels, train_sentences, 
                id2label, k
            )

        # B. Construction des Prompts
        prompts = []
        for idx, (term, sentence) in enumerate(zip(batch_terms, batch_sentences)):
            dynamic_examples = batch_dynamic_examples[idx]
            if dynamic_examples:
                dynamic_examples = f"\nHere are some similar examples to help you:\n{dynamic_examples}\n"
            
            if sentence and len(str(sentence)) > 3:
                base_prompt = f"Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose from: {', '.join(available_labels)}."
            else:
                base_prompt = f"What is the type of the term: '{term}'? Choose from: {', '.join(available_labels)}."

            prompts.append(f"{base_prompt}{dynamic_examples} Answer by only giving the term type.")

        # C. Génération
        with torch.no_grad():
            if is_batched_func:
                batch_responses = gen_func(prompts, model, tokenizer, gen_config)
            else:
                batch_responses = [gen_func(p, model, tokenizer, gen_config) for p in prompts]

        # D. Parsing des Réponses
        for response, actual_idx in zip(batch_responses, batch_labels_idx):
            resp_lower = response.lower() if response else ""
            pred = "unknown"
            
            # Recherche exacte
            for lbl in available_labels:
                if lbl in resp_lower:
                    pred = lbl
                    break
            
            # Recherche premier mot (fallback)
            if pred == "unknown":
                # --- FIX CRASH ---
                split_resp = resp_lower.split()
                if split_resp:
                    first_word = split_resp[0].strip(".,!:")
                    if first_word in available_labels:
                        pred = first_word

            y_pred_all.append(pred)
            y_true_all.append(id2label[actual_idx])
        
        pbar.update(len(batch_indices))
    
    pbar.close()

    # 5. Calcul des Métriques Finales
    calculate_metrics(y_true_all, y_pred_all, model_name=f"{model_name_key} (FT + RAG={use_rag})")
    
    # 6. Nettoyage Mémoire
    del model, base_model, embedder
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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import gc
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import gc
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os

def run_full_error_analysis_matrix(n_samples=100):
    print(f"\n=================================================================================")
    print(f"   ANALYSE D'ERREURS & SAUVEGARDE DES MATRICES ({n_samples} échantillons)")
    print(f"=================================================================================")

    # Crée le dossier pour stocker les images si inexistant
    if not os.path.exists("ConfusionMatrix"):
        os.makedirs("ConfusionMatrix")

    # --- 1. CONFIGURATION DES MODÈLES ---
    # Activez uniquement ceux que vous voulez tester
    MODELS_CONFIG = {
        "Qwen": {"path": "./ft_qwen", "type": "causal"},
        # "Google-Base": {"path": "./ft_google_base", "type": "seq2seq"},
        # "Google-Large": {"path": "./ft_google_large", "type": "seq2seq"},
    }
    
    # --- 2. PRÉPARATION DES DONNÉES (CPU) ---
    print("--> Chargement des données et de l'Embedder RAG sur CPU...")
    # Sélection des indices
    indices = range(min(n_samples, len(text)))
    sample_terms = [text[i] for i in indices]
    sample_sentences = [sentences[i] for i in indices]
    sample_labels_ids = [label[i] for i in indices]
    y_true = [id2label[i] for i in sample_labels_ids]
    
    # [CRITIQUE] On récupère la liste complète des labels possibles (WordNet)
    # pour que la matrice ait toujours la taille 4x4 + Unknown
    labels_list = sorted(list(id2label.values())) 

    embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    
    # Index Train (pour RAG Train)
    t_id2label, t_label2id, tr_terms, tr_labels, tr_sentences = WN_TaskA_TextClf_dataset_builder("train")
    train_emb = embedder.encode(tr_sentences, convert_to_tensor=True).cpu().numpy()
    
    # Index Wiki (pour RAG Wiki) - Subset pour vitesse
    wiki_data = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    wiki_sentences = [txt for txt in wiki_data['text'] if txt.strip() and len(txt.split()) > 5][:2000]
    wiki_emb = embedder.encode(wiki_sentences, convert_to_tensor=True).cpu().numpy()

    # --- 3. BOUCLE SUR LES MODÈLES ---
    for model_key, config in MODELS_CONFIG.items():
        print(f"\n\n################################################")
        print(f"   TRAITEMENT DU MODÈLE : {model_key}")
        print(f"################################################")
        
        # A. Chargement Modèle Base
        global LLM_MODEL 
        LLM_MODEL = model_key 
        
        tokenizer = None
        model = None
        gen_config = None
        gen_func_batched = None

        try:
            # Logique de chargement selon le modèle
            if model_key == "Qwen":
                base_id = "Qwen/Qwen3-4B-Instruct-2507"
                tokenizer = AutoTokenizer.from_pretrained(base_id, padding_side="left")
                if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
                model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")
                gen_config = GenerationConfig(max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
                gen_func_batched = generate_qwen_batched
            else:
                base_id = f"google/flan-t5-{model_key.split('-')[1].lower()}"
                tokenizer = AutoTokenizer.from_pretrained(base_id)
                dtype = torch.float32 if "Small" in model_key else torch.float16
                model = AutoModelForSeq2SeqLM.from_pretrained(base_id, torch_dtype=dtype).to("cuda")
                gen_config = GenerationConfig(max_new_tokens=64)
                if "Small" in model_key:
                    # Wrapper pour Small qui n'a pas de batch natif
                    gen_func_batched = lambda p, m, t, c: [generate_google_simple(x, m, t, c) for x in p]
                else:
                    gen_func_batched = generate_google_batched
            
            # Wrapper unique (transforme liste -> string pour les fonctions existantes)
            def predict_single(p, m, t, c): return gen_func_batched([p], m, t, c)[0]

            # B. Définition des 5 Scénarios
            # Les lambdas permettent de différer l'exécution
            scenarios = {
                "1_Original_LLM": lambda t, s: classify_term_type_with_llm(t, s, available_labels, model, tokenizer, False, predict_single, gen_config),
                "2_RAG_Wikipedia": lambda t, s: classify_term_type_with_rag(t, s, available_labels, model, tokenizer, False, predict_single, gen_config, embedder, wiki_emb, wiki_sentences, 3),
                "3_RAG_TrainSet": lambda t, s: classify_term_type_with_dynamic_few_shot(t, s, available_labels, model, tokenizer, predict_single, gen_config, embedder, train_emb, tr_terms, tr_labels, tr_sentences, t_id2label, 3),
                "4_FineTuned": None, # Sera rempli après chargement adaptateur
                "5_FT_RAG_Train": None 
            }

            results_storage = {}
            
            # C. Inférence PRE-FineTuning (Baselines)
            for name, func in scenarios.items():
                if func is None: continue 
                print(f"   -> Inference: {name}...")
                y_pred = []
                for t, s in zip(sample_terms, sample_sentences):
                    # Nettoyage basique du texte généré
                    y_pred.append(func(t, s).lower().strip().split()[0].strip(".,!:"))
                results_storage[name] = y_pred

            # D. Chargement Adaptateur & Inférence POST-FineTuning
            print(f"   -> Chargement Adaptateur ({config['path']})...")
            try:
                model = PeftModel.from_pretrained(model, config['path'])
                model.eval()
                # On définit les fonctions maintenant que le modèle est Fine-Tuné
                scenarios["4_FineTuned"] = lambda t, s: classify_term_type_with_llm(t, s, available_labels, model, tokenizer, False, predict_single, gen_config)
                scenarios["5_FT_RAG_Train"] = lambda t, s: classify_term_type_with_dynamic_few_shot(t, s, available_labels, model, tokenizer, predict_single, gen_config, embedder, train_emb, tr_terms, tr_labels, tr_sentences, t_id2label, 3)

                for name in ["4_FineTuned", "5_FT_RAG_Train"]:
                    print(f"   -> Inference: {name}...")
                    y_pred = []
                    for t, s in zip(sample_terms, sample_sentences):
                        y_pred.append(scenarios[name](t, s).lower().strip().split()[0].strip(".,!:"))
                    results_storage[name] = y_pred
            except Exception as e:
                print(f"   [!] Impossible de charger l'adaptateur pour {model_key} (Erreur: {e})")

            # E. ANALYSE STATISTIQUE ET VISUELLE
            print(f"\n   --- GÉNÉRATION DES RAPPORTS : {model_key} ---")
            
            for scenario_name, y_pred_raw in results_storage.items():
                # 1. Nettoyage et Alignement des Labels
                y_pred_clean = []
                for p in y_pred_raw:
                    found = False
                    # On cherche si la réponse contient un des labels valides
                    for valid in labels_list:
                        if valid in p:
                            y_pred_clean.append(valid)
                            found = True
                            break
                    if not found:
                        y_pred_clean.append("unknown")
                
                # 2. Calcul des Métriques Leaderboard (Macro F1, etc.)
                # C'est ICI qu'on appelle votre fonction calculate_metrics
                print(f"\n   >>> Scénario : {scenario_name}")
                try:
                    calculate_metrics(y_true, y_pred_clean, model_name=f"{model_key} - {scenario_name}")
                except Exception as met_err:
                    print(f"   (Erreur affichage métriques: {met_err})")

                # 3. Préparation Matrice de Confusion
                # Labels attendus dans la matrice (Classes + Unknown)
                matrix_labels = labels_list + ["unknown"]
                
                # On force 'labels=' pour inclure les classes vides (Adverbe, etc.)
                cm = confusion_matrix(y_true, y_pred_clean, labels=matrix_labels)
                
                # Normalisation en %
                with np.errstate(divide='ignore', invalid='ignore'):
                    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                    cm_percent = np.nan_to_num(cm_percent)

                # 4. Sauvegarde Image (Seaborn Heatmap)
                plt.figure(figsize=(10, 8))
                df_cm = pd.DataFrame(cm_percent, index=matrix_labels, columns=matrix_labels)
                
                # Carte de chaleur (Bleu)
                sns.heatmap(df_cm, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=100)
                
                acc_score = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred_clean[i]]) / len(y_true) * 100
                plt.title(f"Confusion Matrix (%)\nModel: {model_key} | Method: {scenario_name}\nAcc: {acc_score:.1f}%")
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                filename = f"ConfusionMatrix/CM_{model_key}_{scenario_name}.png"
                plt.savefig(filename)
                plt.close()
                print(f"       [Image Sauvegardée] -> {filename}")

                # 5. Affichage Top Erreurs (Console)
                errors = []
                for i in range(len(matrix_labels)): 
                    for j in range(len(matrix_labels)): 
                        if i != j and cm[i, j] > 0:
                            # Ratio par rapport à la classe réelle (ex: 50% des verbes sont faux)
                            ratio = cm[i, j] / cm[i].sum() * 100
                            errors.append((matrix_labels[i], matrix_labels[j], ratio))
                
                errors.sort(key=lambda x: x[2], reverse=True)
                if errors:
                    print("       Top Erreurs (Freq > 0) : " + ", ".join([f"{e[0]}->{e[1]} ({e[2]:.0f}%)" for e in errors[:3]]))

        except Exception as e:
            print(f"ERREUR CRITIQUE SUR {model_key}: {e}")
            import traceback
            traceback.print_exc()
        
        # Nettoyage VRAM entre modèles
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    del embedder, train_emb, wiki_emb
    gc.collect()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def analyze_dataset_stats():
    print(f"\n=======================================================")
    print(f"   ANALYSE STATISTIQUE DU DATASET (Train vs Test)")
    print(f"=======================================================")

    # 1. Chargement des données brutes
    print("--> Chargement des datasets...")
    # On ignore les textes, on veut juste les labels
    _, _, _, train_labels_ids, _ = WN_TaskA_TextClf_dataset_builder("train")
    _, _, _, test_labels_ids, _ = WN_TaskA_TextClf_dataset_builder("test")
    
    # Conversion ID -> Nom (ex: 0 -> 'noun')
    train_labels = [id2label[i] for i in train_labels_ids]
    test_labels = [id2label[i] for i in test_labels_ids]
    
    # 2. Calcul des Stats
    def get_stats(labels, name):
        total = len(labels)
        counts = Counter(labels)
        print(f"\n--- {name} Set (Total: {total}) ---")
        stats = []
        for lbl, count in counts.most_common():
            pct = count / total * 100
            print(f"   {lbl:<10} : {count:>5}  ({pct:>5.2f}%)")
            stats.append({"Label": lbl, "Count": count, "Percentage": pct, "Split": name})
        return stats

    data_stats = []
    data_stats.extend(get_stats(train_labels, "TRAIN"))
    data_stats.extend(get_stats(test_labels, "TEST"))
    
    # 3. Création du DataFrame pour le graphique
    df_stats = pd.DataFrame(data_stats)
    
    # 4. Visualisation (Sauvegarde PNG)
    print("\n--> Génération du graphique de distribution...")
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Graphique en barres côte à côte
    ax = sns.barplot(data=df_stats, x="Label", y="Percentage", hue="Split", palette="viridis")
    
    plt.title("Distribution des Classes (Train vs Test)", fontsize=14)
    plt.ylabel("Pourcentage (%)", fontsize=12)
    plt.xlabel("Type de Terme", fontsize=12)
    plt.legend(title="Dataset")
    
    # Ajout des valeurs au-dessus des barres
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)

    filename = "Dataset_Distribution.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"   [Sauvegardé] -> {filename}")
    print("=======================================================\n")

def run_full_benchmark_and_viz(n_samples=100):
    print(f"\n=================================================================================")
    print(f"   BENCHMARK GLOBAL & VISUALISATION COMBINÉE ({n_samples} échantillons)")
    print(f"=================================================================================")

    if not os.path.exists("GlobalResults"):
        os.makedirs("GlobalResults")

    # --- 1. CONFIGURATION ---
    # Adaptez les chemins vers VOS modèles fine-tunés ici
    MODELS_CONFIG = {
        "Qwen": {"path": "./ft_qwen", "type": "causal"},
        "Google-Small": {"path": "./ft_google_small", "type": "seq2seq"},
        # Ajoutez d'autres modèles ici si vous avez les adaptateurs
        # "Google-Large": {"path": "./ft_google_large", "type": "seq2seq"},
    }
    
    # Stockage global pour les graphiques finaux
    # Structure : global_results[Model][Method] = { "cm": matrice, "acc": float }
    global_results = {m: {} for m in MODELS_CONFIG.keys()}
    
    # --- 2. PRÉPARATION DES DONNÉES (CPU) ---
    print("--> Chargement Data & Embedder...")
    indices = range(min(n_samples, len(text)))
    sample_terms = [text[i] for i in indices]
    sample_sentences = [sentences[i] for i in indices]
    sample_labels_ids = [label[i] for i in indices]
    y_true = [id2label[i] for i in sample_labels_ids]
    
    # Liste fixe des labels pour avoir des matrices de même taille partout
    labels_list = sorted(list(id2label.values()))
    matrix_labels = labels_list + ["unknown"]

    embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    t_id2label, t_label2id, tr_terms, tr_labels, tr_sentences = WN_TaskA_TextClf_dataset_builder("train")
    train_emb = embedder.encode(tr_sentences, convert_to_tensor=True).cpu().numpy()
    
    wiki_data = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    wiki_sentences = [txt for txt in wiki_data['text'] if txt.strip() and len(txt.split()) > 5][:2000]
    wiki_emb = embedder.encode(wiki_sentences, convert_to_tensor=True).cpu().numpy()

    # --- 3. BOUCLE D'INFÉRENCE ---
    for model_key, config in MODELS_CONFIG.items():
        print(f"\n>>> Traitement Modèle : {model_key}")
        
        # A. Chargement Modèle Base
        # (Note: Je simplifie la logique ici pour la lisibilité, assurez-vous que les imports sont bons)
        tokenizer = None; model = None; gen_config = None; gen_func_batched = None
        
        try:
            if config["type"] == "causal": # Qwen
                base_id = "Qwen/Qwen3-4B-Instruct-2507"
                tokenizer = AutoTokenizer.from_pretrained(base_id, padding_side="left")
                if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
                model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")
                gen_config = GenerationConfig(max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
                gen_func_batched = generate_qwen_batched
            else: # T5
                base_id = f"google/flan-t5-{model_key.split('-')[1].lower()}"
                tokenizer = AutoTokenizer.from_pretrained(base_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(base_id, torch_dtype=torch.float32).to("cuda")
                gen_config = GenerationConfig(max_new_tokens=64)
                # Petit hack pour T5-Small qui n'a pas de batch natif dans votre code original
                if "Small" in model_key: gen_func_batched = lambda p, m, t, c: [generate_google_simple(x, m, t, c) for x in p]
                else: gen_func_batched = generate_google_batched

            def predict_single(p, m, t, c): return gen_func_batched([p], m, t, c)[0]

            # B. Définition des Scénarios
            scenarios = {
                "Zero-Shot": lambda t, s: classify_term_type_with_llm(t, s, available_labels, model, tokenizer, False, predict_single, gen_config),
                "RAG Wiki": lambda t, s: classify_term_type_with_rag(t, s, available_labels, model, tokenizer, False, predict_single, gen_config, embedder, wiki_emb, wiki_sentences, 3),
                "RAG Train": lambda t, s: classify_term_type_with_dynamic_few_shot(t, s, available_labels, model, tokenizer, predict_single, gen_config, embedder, train_emb, tr_terms, tr_labels, tr_sentences, t_id2label, 3),
                "Fine-Tuned": None,
                "FT + RAG": None 
            }

            # C. Inférence Base (Avant chargement adaptateur)
            for name, func in scenarios.items():
                if func is None: continue 
                print(f"   -> {name}...")
                y_pred = [func(t, s).lower().strip().split()[0].strip(".,!:") for t, s in zip(sample_terms, sample_sentences)]
                
                # Nettoyage et Calcul CM
                y_pred_clean = [l if l in labels_list else "unknown" for l in y_pred]
                cm = confusion_matrix(y_true, y_pred_clean, labels=matrix_labels)
                acc = accuracy_score(y_true, y_pred_clean) * 100
                
                # Normalisation
                with np.errstate(divide='ignore', invalid='ignore'):
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                    cm_norm = np.nan_to_num(cm_norm)
                
                global_results[model_key][name] = {"cm": cm_norm, "acc": acc}

            # D. Chargement Adaptateur & Inférence Fine-Tuned
            if os.path.exists(config['path']):
                print(f"   -> Chargement Adaptateur {config['path']}...")
                model = PeftModel.from_pretrained(model, config['path'])
                model.eval()
                
                scenarios["Fine-Tuned"] = lambda t, s: classify_term_type_with_llm(t, s, available_labels, model, tokenizer, False, predict_single, gen_config)
                scenarios["FT + RAG"] = lambda t, s: classify_term_type_with_dynamic_few_shot(t, s, available_labels, model, tokenizer, predict_single, gen_config, embedder, train_emb, tr_terms, tr_labels, tr_sentences, t_id2label, 3)

                for name in ["Fine-Tuned", "FT + RAG"]:
                    print(f"   -> {name}...")
                    y_pred = [scenarios[name](t, s).lower().strip().split()[0].strip(".,!:") for t, s in zip(sample_terms, sample_sentences)]
                    y_pred_clean = [l if l in labels_list else "unknown" for l in y_pred]
                    cm = confusion_matrix(y_true, y_pred_clean, labels=matrix_labels)
                    acc = accuracy_score(y_true, y_pred_clean) * 100
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                        cm_norm = np.nan_to_num(cm_norm)

                    global_results[model_key][name] = {"cm": cm_norm, "acc": acc}
            else:
                print(f"   [!] Adaptateur introuvable pour {model_key}, skip FT.")

        except Exception as e:
            print(f"ERREUR {model_key}: {e}")
        
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # --- 4. GÉNÉRATION DES GRAPHIQUES COMBINÉS ---
    print("\n>>> Génération des visualisations globales...")
    
    methods_order = ["Zero-Shot", "RAG Wiki", "RAG Train", "Fine-Tuned", "FT + RAG"]
    models_order = list(MODELS_CONFIG.keys())
    
    # A. La "SUPER-GRILLE" de Matrices de Confusion
    n_rows = len(models_order)
    n_cols = len(methods_order)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)
    if n_rows == 1: axes = np.array([axes]) # Fix dimensions si 1 seul modèle
    if n_cols == 1: axes = axes[:, np.newaxis] # Fix dimensions si 1 seule méthode
    if len(axes.shape) == 1: axes = axes.reshape(n_rows, n_cols) # Safety

    for i, mod in enumerate(models_order):
        for j, meth in enumerate(methods_order):
            ax = axes[i, j]
            
            if meth in global_results[mod]:
                data = global_results[mod][meth]
                sns.heatmap(data["cm"], annot=True, fmt=".0f", cmap="Blues", 
                            vmin=0, vmax=100, cbar=False, ax=ax,
                            xticklabels=matrix_labels, yticklabels=matrix_labels, annot_kws={"size": 8})
                ax.set_title(f"Acc: {data['acc']:.1f}%", fontsize=10, color='darkgreen')
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                
            # Titres colonnes (Méthodes)
            if i == 0: ax.text(0.5, 1.15, meth, transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')
            # Titres lignes (Modèles)
            if j == 0: ax.set_ylabel(f"{mod}\nTrue Label", fontsize=11, fontweight='bold')
            else: ax.set_ylabel("")
            
            if i == n_rows - 1: ax.set_xlabel("Predicted")
            else: ax.set_xlabel("")

    plt.suptitle("Comparaison Complète : Matrices de Confusion par Modèle & Méthode", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("GlobalResults/All_Models_Confusion_Matrix.png", dpi=150)
    plt.close()
    print("   -> [1] GlobalResults/All_Models_Confusion_Matrix.png généré.")

    # B. Le Heatmap de Résumé (Accuracy)
    summary_data = []
    for mod in models_order:
        row = []
        for meth in methods_order:
            if meth in global_results[mod]:
                row.append(global_results[mod][meth]["acc"])
            else:
                row.append(0)
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data, index=models_order, columns=methods_order)
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(df_summary, annot=True, fmt=".1f", cmap="RdYlGn", vmin=40, vmax=90, cbar_kws={'label': 'Accuracy (%)'})
    plt.title("Synthèse des Performances (Accuracy Global)", fontsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("GlobalResults/Accuracy_Summary_Heatmap.png", dpi=150)
    plt.close()
    print("   -> [2] GlobalResults/Accuracy_Summary_Heatmap.png généré.")

if __name__ == "__main__":
    #print(f"=====================K = {0} ========================")
    #run_classification("classify_term_type_with_llm", k=0)
    #for k in range(1,6):
    #    print(f"\n\n================== K = {k} ==================")
    #    run_classification("classify_term_type_with_dynamic_few_shot", k)
    #for k in range(1,11):
    #    run_classification("classify_term_type_with_rag", k)
    #adapter_path = run_finetuning("Qwen", output_dir="./ft_qwen")
    #evaluate_finetuned_model("Qwen", "./ft_qwen", use_rag=False, k=0, batch_size=32)
    #for i in range(1,6):
    #    print(f"\n\n================== K = {i} ==================")
    #    evaluate_finetuned_model("Qwen", "./ft_qwen", use_rag=True, k=i, batch_size=32)
    #run_classification("classify_term_type_with_llm", k=0)

    #path_ft = "./ft_google_small"
    
    # Lancement du benchmark
    #benchmark_all_methods(path_ft, n_samples=10)

    # Remplacez par votre dossier d'adaptateur
    #path_ft = "./ft_qwen" 
    
    # Lance l'analyse sur 50 phrases au hasard (rapide)
    #perform_error_analysis(path_ft, n_samples=100)

    #run_full_error_analysis_matrix(n_samples=100)

    #analyze_dataset_stats()

    run_full_benchmark_and_viz(n_samples=200)