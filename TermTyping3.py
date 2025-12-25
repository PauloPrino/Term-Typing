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

        if n_gpus > 1:
            generated_ids = llm_model.module.generate(
                **batch_inputs, 
                max_new_tokens=generation_cfg.max_new_tokens,
                pad_token_id=tokenizer_model.pad_token_id
            )
        else:
            generated_ids = llm_model.generate(
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
        model_to_run = llm_model.module if n_gpus > 1 else llm_model
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
            generated_ids = llm_model.generate(
                inputs.input_ids,
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

def format_dataset_for_training(terms, sentences, labels_ids, id2label, tokenizer, model_type):
    available_labels = list(id2label.values())
    formatted_data = []

    for i in range(len(terms)):
        term = terms[i]
        sentence = sentences[i]
        label_text = id2label[labels_ids[i]]

        if sentence and len(str(sentence)) > 3:
            prompt = f"Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose from: {', '.join(available_labels)}. Answer by only giving the term type."
        else:
            prompt = f"What is the type of the term: '{term}'? Choose from: {', '.join(available_labels)}. Answer by only giving the term type."

        if model_type == "seq2seq": # T5
            text = f"{prompt} {label_text}"
        else: # CausalLM (Qwen)
            text = f"User: {prompt}\nAssistant: {label_text}<|endoftext|>"

        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)

def run_finetuning(model_name_key, output_dir="./finetuned_model"):
    print(f"--- Démarrage du Fine-Tuning pour : {model_name_key} ---")
    
    train_id2label, train_label2id, train_terms, train_labels, train_sentences = WN_TaskA_TextClf_dataset_builder("train")
    
    if model_name_key == "Google-Small":
        model_id = "google/flan-t5-small"
        model_type = "seq2seq"
        target_modules = ["q", "v"] 
        bnb_config = None 
    elif model_name_key == "Google-Large":
        model_id = "google/flan-t5-large"
        model_type = "seq2seq"
        target_modules = ["q", "v"]
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif model_name_key == "Qwen":
        model_id = "Qwen/Qwen3-4B-Instruct-2507"
        model_type = "causal"
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # --- FIX P100: Compute Dtype Float16 (Safe) ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_use_double_quant=False,
        )
    else:
        raise ValueError("Modèle inconnu.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cuda")
    else:
        tokenizer.pad_token = tokenizer.eos_token
        
        # --- FIX: Initial Load in Float16 ---
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            torch_dtype=torch.float16 
        )
        
        # --- FIX: Overwrite Internal Config Metadata ---
        # Stop accelerate/PEFT from assuming BF16 because the JSON says so.
        model.config.torch_dtype = torch.float16 
        model.config.use_cache = False

    train_dataset = format_dataset_for_training(
        train_terms, train_sentences, train_labels, train_id2label, tokenizer, model_type
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM" if model_type == "seq2seq" else "CAUSAL_LM"
    )
    
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, peft_config)
    
    # --- FIX ULTIME: BRUTE FORCE FLOAT32 CONVERSION ---
    # Sur P100, on ne peut ABSOLUMENT PAS avoir de BF16.
    # Les paramètres "trainable" (LoRA) sont ceux qui posent problème dans l'optimizer.
    # On les force en float32 (Standard QLoRA = 4bit Base + Float32 Adapters).
    print(">>> [Sécurité P100] Vérification et conversion FORCEE des paramètres...")
    
    count_converted = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Si c'est un adapter ou une norme trainable
            # On force le float32 pour la stabilité numérique et éviter le crash BF16
            if param.dtype != torch.float32:
                print(f" -> Conversion {name} de {param.dtype} vers float32")
                param.data = param.data.to(torch.float32)
                count_converted += 1
                
    print(f">>> {count_converted} paramètres critiques convertis en float32.")

    model.print_trainable_parameters()

    from trl import __version__ as trl_version
    from packaging import version
    
    # --- Training Args: STRICT FP16 (Classic AMP) ---
    sft_config_kwargs = {
        "output_dir": f"{output_dir}_checkpoints",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "logging_steps": 10,
        "num_train_epochs": 1,
        "save_strategy": "epoch",
        "fp16": False,   # Active le scaler FP16 classique (supporté par P100)
        "bf16": False,  # DESACTIVE le mode BF16 (Crash assuré sur P100)
        "report_to": "none",
    }
    
    training_args = SFTConfig(**sft_config_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": training_args,
    }

    current_ver = version.parse(trl_version)
    if current_ver >= version.parse("0.25.0"):
        print(f"Détection TRL Récent ({trl_version})")
        training_args.dataset_text_field = "text"
        training_args.max_seq_length = 256
        trainer_kwargs["processing_class"] = tokenizer
    else:
        print(f"Détection TRL Ancien ({trl_version})")
        trainer_kwargs["tokenizer"] = tokenizer
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = 256

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    
    print(f"Sauvegarde de l'adaptateur dans {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_dir

def evaluate_finetuned_model(model_name_key, adapter_path):
    print(f"--- Chargement du modèle Fine-Tuné ({model_name_key}) pour évaluation ---")
    
    if model_name_key == "Google-Small":
        base_model_id = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float32).to("cuda")
        gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)
        gen_func = generate_google_simple 
        
    elif model_name_key == "Qwen":
        base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        # Load in float16 to match what we did in training context
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        gen_config = GenerationConfig(max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        gen_func = generate_qwen_batched
        
    else: 
        base_model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
        gen_config = GenerationConfig(max_new_tokens=64)
        gen_func = generate_google_batched

    print(f"Chargement des poids LoRA depuis {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print("Lancement de la boucle de prédiction...")
    
    available_labels = list(id2label.values())
    correct_predictions = 0
    total_predictions = 0
    
    pbar = tqdm(zip(text, sentences, label), total=len(text))
    
    for test_term, test_sentence, actual_label_idx in pbar:
        actual_label = id2label[actual_label_idx]
        
        if test_sentence and len(str(test_sentence)) > 3:
            prompt = f"Given the term '{test_term}' in the sentence '{test_sentence}', what is the type of the term? Choose from: {', '.join(available_labels)}. Answer by only giving the term type."
        else:
            prompt = f"What is the type of the term: '{test_term}'? Choose from: {', '.join(available_labels)}. Answer by only giving the term type."

        with torch.no_grad():
            if "batched" in gen_func.__name__:
                 res = gen_func([prompt], model, tokenizer, gen_config)
            else:
                 res = gen_func(prompt, model, tokenizer, gen_config)
            
            generated_text = res[0] if isinstance(res, list) else res

        generated_text_lower = generated_text.lower()
        predicted = "unknown"
        
        for lbl in available_labels:
            if lbl in generated_text_lower:
                predicted = lbl
                break
        
        if predicted == actual_label:
            correct_predictions += 1
        total_predictions += 1
        
        pbar.set_postfix({"Acc": f"{correct_predictions/total_predictions*100:.2f}%"})

    print(f"Final Accuracy (Fine-Tuned): {correct_predictions/total_predictions*100:.2f}%")

if __name__ == "__main__":
    #for k in range(3,11):
    #    run_classification("classify_term_type_with_dynamic_few_shot", k)
    adapter_path = run_finetuning("Google-Large", output_dir="./ft_google_large")
    evaluate_finetuned_model("Google-Large", adapter_path)