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


root_path = "/home/infres/pprin-23/LLM/TermTyping"

LLM_MODEL = "Google-Small"

"""## 1. Load WordNet Data"""

def WN_TaskA_TextClf_dataset_builder(train_test: str):
    if train_test == "train":
        json_file = root_path + "/WordNet/A.1(FS)_WordNet_Train.json"
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

    elif train_test == "test":
        # Pour le test, on doit charger DEUX fichiers et les fusionner
        data_file = root_path + "/WordNet/A.1(FS)_WordNet_Test.json"
        gt_file = root_path + "/WordNet/A.1(FS)_WordNet_Test_GT.json"

        with open(data_file, 'r', encoding='utf-8') as f_data:
            test_data = json.load(f_data)
        with open(gt_file, 'r', encoding='utf-8') as f_gt:
            gt_data = json.load(f_gt)

        # Créer un dictionnaire pour retrouver rapidement le type via l'ID
        # Note: Dans le GT, le type est souvent une liste ["noun"], on prend le premier élément
        gt_lookup = {}
        for item in gt_data:
            label = item['type']
            if isinstance(label, list):
                label = label[0]
            gt_lookup[item['ID']] = label

        # Fusionner les informations (term/sentence du fichier Test + type du fichier GT)
        data = []
        for item in test_data:
            # On ne garde que ceux qui ont une vérité terrain correspondante
            if item['ID'] in gt_lookup:
                new_item = item.copy()
                new_item['type'] = gt_lookup[item['ID']]
                data.append(new_item)

    # Le reste de la fonction reste identique et fonctionne maintenant pour train ET test
    types = set()
    #de-duplicate
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
#print(WN_TaskA_TextClf_dataset_builder())

# --- MODIFICATION POUR 2 GPUs ---
# On détecte si on a plusieurs GPUs
device = "cuda" if torch.cuda.is_available() else "cpu"
n_gpus = torch.cuda.device_count()
print(f"Utilisation de {n_gpus} GPUs !")

if LLM_MODEL == "Qwen":
    llm_name_qwen = "Qwen/Qwen3-4B-Instruct-2507"
    
    tokenizer_qwen = AutoTokenizer.from_pretrained(llm_name_qwen, padding_side="left")
    tokenizer_qwen.use_default_system_prompt = False
    tokenizer_qwen.pad_token_id = tokenizer_qwen.eos_token_id

    # 1. On charge d'abord sur le GPU principal (pas de device_map ici !)
    llm_qwen = AutoModelForCausalLM.from_pretrained(
        llm_name_qwen,
        torch_dtype=torch.float16
    ).to(device) # On envoie vers CUDA

    # 2. Si on a plusieurs GPUs, on active le DataParallel
    if n_gpus > 1:
        llm_qwen = torch.nn.DataParallel(llm_qwen)

    llm_qwen.eval()
    
    # Configuration de génération inchangée
    generation_config_qwen = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer_qwen.eos_token_id,
        pad_token_id=tokenizer_qwen.pad_token_id,
    )

    # 3. Fonction de génération par BATCH (beaucoup plus rapide)
    def generate_qwen_batched(prompts, llm_model, tokenizer_model, generation_cfg):
        turns_batch = [[{"role": "user", "content": p}] for p in prompts]
        
        # 1. OPTIMISATION : On demande au template de nous rendre du TEXTE (str), pas des IDs.
        # Cela nous donne le prompt formaté avec les balises système, user, etc.
        text_batch = [
            tokenizer_model.apply_chat_template(
                turn, 
                tokenize=False, # <--- C'est ici que ça change
                add_generation_prompt=True
            ) 
            for turn in turns_batch
        ]
        
        # 2. On passe la liste de textes bruts directement au tokenizer
        # C'est la méthode "__call__" recommandée par le warning.
        # Elle gère tokenization + padding + attention_mask en une seule opération optimisée en Rust/C++
        batch_inputs = tokenizer_model(
            text_batch, 
            padding=True, 
            return_tensors="pt"
        ).to(device)

        # 3. Génération (Le masque d'attention est inclus automatiquement dans batch_inputs)
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

        # Décodage
        outputs = []
        input_len = batch_inputs["input_ids"].shape[-1]
        for i in range(len(generated_ids)):
            output_ids = generated_ids[i][input_len:]
            content = tokenizer_model.decode(output_ids, skip_special_tokens=True)
            outputs.append(content.strip())
            
        return outputs
elif LLM_MODEL == "Google-Large":
    # On prend le modèle LARGE car le SMALL est trop faible pour cette tâche (comme discuté avant)
    llm_name_google = "google/flan-t5-large" 
    
    # CORRECTION CRITIQUE ICI :
    # 1. On enlève padding_side="left" (T5 préfère droite par défaut)
    tokenizer_google = AutoTokenizer.from_pretrained(llm_name_google)
    tokenizer_google.use_default_system_prompt = False
    
    # 2. On NE remplace PAS le pad_token par eos_token
    # T5 a déjà un vrai pad_token (<pad>), on le laisse faire son travail.
    # (Supprimez la ligne tokenizer_google.pad_token_id = tokenizer_google.eos_token_id)

    llm_google = AutoModelForSeq2SeqLM.from_pretrained(
        llm_name_google,
        torch_dtype=torch.float16
    ).to(device)

    # Note sur le DataParallel avec .generate() :
    # DataParallel ne parallélise pas nativement la fonction .generate().
    # Votre code actuel (llm_model.module.generate) s'exécute sur UN SEUL GPU.
    # Pour l'instant, gardez ça pour que ça marche, mais sachez que vous n'avez pas un vrai gain x2.
    if n_gpus > 1:
        llm_google = torch.nn.DataParallel(llm_google)
        
    llm_google.eval()

    generation_config_google = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        # On utilise les tokens par défaut du tokenizer
        eos_token_id=tokenizer_google.eos_token_id,
        pad_token_id=tokenizer_google.pad_token_id,
    )

    def generate_google_batched(prompts, llm_model, tokenizer_model, generation_cfg):
        # Le tokenizer va utiliser le padding par défaut (DROITE pour T5)
        inputs = tokenizer_model(prompts, return_tensors="pt", padding=True).to(device)
        
        # On récupère le modèle de base pour le generate
        model_to_run = llm_model.module if n_gpus > 1 else llm_model
        
        generated_ids = model_to_run.generate(
            **inputs, 
            max_new_tokens=generation_cfg.max_new_tokens
        )
        
        outputs = tokenizer_model.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs
elif LLM_MODEL == "Google-Small":
    print("--- Mode: Google Flan-T5 Small (SAFE) ---")

    llm_name_google = "google/flan-t5-small"

    # 1️⃣ Tokenizer STANDARD (pas de padding forcé)
    tokenizer_google = AutoTokenizer.from_pretrained(llm_name_google)

    # 2️⃣ Modèle en float32 (important pour Small)
    llm_google = AutoModelForSeq2SeqLM.from_pretrained(
        llm_name_google,
        torch_dtype=torch.float32
    ).to(device)

    llm_google.eval()

    # 3️⃣ Génération courte et déterministe
    generation_config_google = GenerationConfig(
        max_new_tokens=20,
        do_sample=False
    )

    import re

    def generate_google_small_batched(prompts, llm_model, tokenizer_model, generation_cfg):
        """
        Version SAFE pour Flan-T5-Small :
        - pas de padding
        - 1 prompt à la fois (API batch-compatible)
        - nettoyage robuste pour le parsing
        """
        outputs = []

        for prompt in prompts:
            inputs = tokenizer_model(
                prompt,
                return_tensors="pt"
            ).to(llm_model.device)

            with torch.no_grad():
                generated_ids = llm_model.generate(
                    inputs.input_ids,
                    max_new_tokens=generation_cfg.max_new_tokens
                )

            text = tokenizer_model.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).lower()

            # 🔧 NETTOYAGE CRITIQUE
            text = re.sub(r"[^a-z]", " ", text)
            text = " ".join(text.split())

            outputs.append(text)

        return outputs



"""## Classification tasks"""

def classify_term_type_with_llm(term, sentence, labels, llm_model, tokenizer_model, few_shot_prompting, active_generate_func, generation_cfg):
    """
    Classifies the type of a term within a sentence using a Causal Language Model
    via prompt engineering.
    """
    # Construct the prompt to guide the LLM towards the desired classification.
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

    # Generate a response from the model
    with torch.no_grad():
      generated_text = active_generate_func(prompt, llm_model, tokenizer_model, generation_cfg)

    # Simple heuristic to extract the label from the generated text.
    generated_text_lower = generated_text.lower()
    for label_name in labels:
        if label_name in generated_text_lower:
            return label_name

    # Fallback: if no direct match, try matching the first word of the generated text.
    first_word_generated = generated_text_lower.split(" ")[0]
    if first_word_generated in labels:
        return first_word_generated

    return "Unknown or no clear label generated"

# Set the active LLM, tokenizer, generation config and generate function
# This block allows you to easily switch between Qwen and Google for evaluation

def run_classification(classification_task, k):
    if LLM_MODEL == "Google-Large":
        active_llm_model = llm_google
        active_tokenizer_model = tokenizer_google
        active_generation_config = generation_config_google
        # On pointe directement vers la fonction batched
        active_generate_func = generate_google_batched 
        print("LLM used: Google Flan T5 Large")
    if LLM_MODEL == "Google-Small":
        active_llm_model = llm_google
        active_tokenizer_model = tokenizer_google
        active_generation_config = generation_config_google
        batch_generate_func = generate_google_small_batched
        print("LLM used: Google Flan T5 Small (SAFE)")

    elif LLM_MODEL == "Qwen":
        active_llm_model = llm_qwen
        active_tokenizer_model = tokenizer_qwen
        active_generation_config = generation_config_qwen
        # On pointe directement vers la fonction batched
        active_generate_func = generate_qwen_batched
        print("LLM used: Qwen3 4B Instruct")
    # ----------------------

    print(f"Running predictions on Test Set for the classification task {classification_task}")

    if classification_task == "classify_term_type_with_rag":
        # 2. Initialize the Embedding Model
        # This model is small, fast, and effective for sentence similarity
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Use external Wikipedia data (Wikitext-103)
        # Only use this if you don't want to use the training set.

        print("Loading Wikitext-103 (this may take a minute)...")
        # Load a small subset (validation split) to keep it fast for this demo
        wiki_data = load_dataset("wikitext", "wikitext-103-v1", split="validation")

        # Extract non-empty lines to simulate a sentence store
        wikidata_sentences = [text for text in wiki_data['text'] if text.strip() and len(text.split()) > 5]

        print(f"RAG Knowledge Base loaded with {len(wikidata_sentences)} Wikipedia sentences.")
        print("Encoding Wikidata sentences... (this may take a while for large datasets)")
        # Encode all sentences in the knowledge base into vectors
        wikidata_embeddings = embedder.encode(wikidata_sentences, convert_to_tensor=True).cpu().numpy()
        print("Encoding complete.")
    elif classification_task == "classify_term_type_with_dynamic_few_shot":
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # CHANGEZ CETTE LIGNE : Renommez id2label en train_id2label, etc.
        train_id2label, train_label2id, train_terms, train_labels, train_sentences = WN_TaskA_TextClf_dataset_builder("train")

        print("Encoding Train sentences for Dynamic Few-Shot...")

        # On encode les phrases d'entraînement pour trouver les plus ressemblantes syntaxiquement
        train_embeddings = embedder.encode(train_sentences, convert_to_tensor=True).cpu().numpy()
        print("Encoding complete.")

    correct_predictions = 0
    total_predictions = 0
    
    # Sélectionnez la fonction batched appropriée
    if LLM_MODEL == "Qwen":
        batch_generate_func = generate_qwen_batched
    elif LLM_MODEL == "Google-Small":
        batch_generate_func = generate_google_small_batched
    elif LLM_MODEL == "Google-Large":
        batch_generate_func = generate_google_batched

    BATCH_SIZE = 1 # On traite 16 phrases à la fois (8 par GPU)
    
    # On prépare les données
    all_indices = list(range(len(text)))
    
    # Barre de progression par batch
    pbar = tqdm(total=len(text))
    
    for i in range(0, len(text), BATCH_SIZE):
        batch_indices = all_indices[i : i + BATCH_SIZE]
        
        batch_terms = [text[j] for j in batch_indices]
        batch_sentences = [sentences[j] for j in batch_indices]
        batch_labels_idx = [label[j] for j in batch_indices]
        
        # ... Début de la boucle "for i in range(0, len(text), BATCH_SIZE):" ...

        prompts = []
        
        # --- MODIFICATION START : Recherche groupée pour utiliser les 10 CPUs ---
        batch_dynamic_examples = []
        if classification_task == "classify_term_type_with_dynamic_few_shot":
            # On appelle la fonction UNE FOIS pour les 16 phrases
            # Les 10 CPUs vont travailler ensemble sur ce calcul matriciel
            batch_dynamic_examples = get_dynamic_few_shot_examples_batched(
                batch_sentences, batch_terms, 
                embedder, train_embeddings, train_terms, train_labels, train_sentences, 
                train_id2label,k
            )
        else:
            batch_dynamic_examples = [""] * len(batch_terms)

        # Construction des prompts (Instantané maintenant)
        for idx, (term, sentence) in enumerate(zip(batch_terms, batch_sentences)):
            dynamic_examples = batch_dynamic_examples[idx]
            
            prompt = f"""Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose from: {', '.join(available_labels)}.
            {dynamic_examples}
            Answer by only giving the term type."""
            prompts.append(prompt)
            
        # ... Suite du code (Génération GPU) ...

        # 2. Génération PARALLELE (C'est là que les 2 GPUs travaillent)
        batch_responses = batch_generate_func(prompts, active_llm_model, active_tokenizer_model, active_generation_config)

        # 3. Analyse des réponses
        for response, actual_idx in zip(batch_responses, batch_labels_idx):
            # Votre logique de parsing
            resp_lower = response.lower()
            pred = "unknown"
            for lbl in available_labels:
                if lbl in resp_lower:
                    pred = lbl
                    break
            
            actual_txt = id2label[actual_idx]
            if pred == actual_txt:
                correct_predictions += 1
            total_predictions += 1
            
        pbar.update(len(batch_indices))
        current_acc = correct_predictions / total_predictions
        pbar.set_postfix({"Accuracy": f"{current_acc * 100:.2f}%"})

    accuracy = correct_predictions / total_predictions
    print(f"\nFinal Accuracy for the classification task {classification_task} and k={k}: {accuracy * 100:.2f}%")

"""## Chain of Thought

## RAG

In the case where we don't have a sentence but just the term we search for occurences of this term in the dataset directly (not with the embedding, not with similarity) and we take as a sentence an occurence of it with the previous words and newt words with which it appears with a certain probability distribution deciding how many terms before and after we have and also how many sentences we take/build.

## RAG on Wikipedia
"""

def get_rag_context(query_sentence, embedder, wikidata_embeddings, wikidata_sentences, k):
    """
    Retrieves the top-k most similar sentences from the wikidata_sentences
    based on the query_sentence.
    """
    if not query_sentence:
        return ""

    # Embed the query sentence
    query_embedding = embedder.encode([query_sentence], convert_to_tensor=True).cpu().numpy()

    # Calculate cosine similarity between query and all wikidata sentences
    # valid range: [-1, 1], higher is more similar
    similarities = cosine_similarity(query_embedding, wikidata_embeddings)

    # Get the indices of the top-k most similar sentences
    # argsort sorts in ascending order, so we take the last k elements and reverse them
    top_k_indices = similarities[0].argsort()[-k:][::-1]

    # Retrieve the actual sentences
    retrieved_contexts = [wikidata_sentences[idx] for idx in top_k_indices]

    # Join them if k > 1
    return " ".join(retrieved_contexts)

def classify_term_type_with_rag(term, sentence, labels, llm_model, tokenizer_model, few_shot_prompting, active_generate_func, generation_cfg, embedder, wikidata_embeddings, wikidata_sentences, k):
    """
    Classifies term type using RAG to inject similar context.
    """
    # 1. Retrieve Context
    # We find a sentence in our knowledge base that is semantically similar to the input
    retrieved_context = get_rag_context(sentence, embedder, wikidata_embeddings, wikidata_sentences,k)

    context_block = ""
    if retrieved_context:
        context_block = f"Here are other similar sentences to help you: {retrieved_context}\n"
        #print(context_block)

    # 2. Construct Prompt with RAG
    # We provide the retrieved sentence as a reference "Context" or "Similar usage"
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

    # 3. Generate
    with torch.no_grad():
      generated_text = active_generate_func(prompt, llm_model, tokenizer_model, generation_cfg)

    # Simple heuristic to extract the label from the generated text.
    generated_text_lower = generated_text.lower()
    for label_name in labels:
        if label_name in generated_text_lower:
            return label_name

    # Fallback: if no direct match, try matching the first word of the generated text.
    first_word_generated = generated_text_lower.split(" ")[0]
    if first_word_generated in labels:
        return first_word_generated

    return "Unknown or no clear label generated"

"""## RAG on the Train set"""

def get_dynamic_few_shot_examples(query_sentence, query_term, embedder, train_embeddings, train_terms, train_labels, train_sentences, id2label, k):
    """
    Récupère 3 exemples du Train Set.
    Si la phrase est vide, on utilise le TERME pour trouver des phrases
    qui parlent de ce concept dans le train set.
    """
    # Si la phrase est vide ou trop courte, on utilise le terme comme requête
    search_query = query_sentence if len(str(query_sentence)) > 3 else query_term

    if not search_query:
        return ""

    query_embedding = embedder.encode([str(search_query)], convert_to_tensor=True).cpu().numpy()

    # Calcul de similarité
    similarities = cosine_similarity(query_embedding, train_embeddings)

    # Top K
    top_k_indices = similarities[0].argsort()[-k:][::-1]

    examples_text = ""
    for idx in top_k_indices:
        ex_term = train_terms[idx]
        ex_sent = train_sentences[idx]
        # Gestion propre des labels (ids ou strings)
        raw_label = train_labels[idx]
        ex_label = id2label[raw_label] if isinstance(raw_label, int) else raw_label

        # On ne garde l'exemple que s'il a du sens (évite de récupérer des exemples vides)
        if len(str(ex_sent)) > 3:
            examples_text += f"- term: {ex_term}, sentence: {ex_sent}, answer: {ex_label}\n"

    # Si on a rien trouvé de pertinent (ex: que des phrases vides), on met un fallback
    if not examples_text:
         examples_text = "- term: apple, sentence: I ate an apple, answer: noun\n"

    return examples_text

def get_dynamic_few_shot_examples_batched(batch_sentences, batch_terms, embedder, train_embeddings, train_terms, train_labels, train_sentences, id2label, k):
    """
    Version optimisée CPU : Utilise les 10 cœurs pour calculer la similarité 
    sur tout le batch d'un coup (Vectorisation).
    """
    # 1. Préparation des requêtes pour tout le batch
    queries = []
    for s, t in zip(batch_sentences, batch_terms):
        queries.append(str(s) if len(str(s)) > 3 else str(t))

    if not queries:
        return [""] * len(batch_sentences)

    # 2. Encodage VECTORIEL (Rapide)
    # Numpy va utiliser vos 10 CPUs ici pour les opérations matricielles
    query_embeddings = embedder.encode(queries, convert_to_tensor=True).cpu().numpy()

    # 3. Calcul de similarité GÉANT
    # C'est ICI que le gain x10 se fait : une seule grosse multiplication matricielle
    similarities = cosine_similarity(query_embeddings, train_embeddings)

    batch_examples_text = []
    
    # 4. Extraction des résultats
    for i in range(len(queries)):
        top_k_indices = similarities[i].argsort()[-k:][::-1]
        
        examples_text = ""
        for idx in top_k_indices:
            ex_term = train_terms[idx]
            ex_sent = train_sentences[idx]
            raw_label = train_labels[idx]
            # Gestion int/str pour le label
            ex_label = id2label[raw_label] if isinstance(raw_label, int) else raw_label
            
            if len(str(ex_sent)) > 3:
                examples_text += f"- term: {ex_term}, sentence: {ex_sent}, answer: {ex_label}\n"
        
        if not examples_text:
             examples_text = "- term: apple, sentence: I ate an apple, answer: noun\n"
        
        batch_examples_text.append(examples_text)

    return batch_examples_text

def classify_term_type_with_dynamic_few_shot(term, sentence, labels, llm_model, tokenizer_model, active_generate_func, generation_cfg, embedder, train_embeddings, train_terms, train_labels, train_sentences, id2label, k):

    # 1. Récupération intelligente (on passe aussi le terme)
    dynamic_examples = get_dynamic_few_shot_examples(sentence, term, embedder, train_embeddings, train_terms, train_labels, train_sentences, id2label, k)

    # 2. Construction du prompt
    prompt = f"""
Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose the term type from: {', '.join(labels)}.

Here are some examples of similar cases to help you:
{dynamic_examples}

Answer by only giving the term type and not explaining.
"""

    # 3. Génération via la fonction WRAPPER
    generated_text = active_generate_func(prompt, llm_model, tokenizer_model, generation_cfg)

    # 4. Parsing (inchangé)
    generated_text_lower = generated_text.lower()

    # Recherche exacte
    for label_name in labels:
        if label_name in generated_text_lower:
            return label_name

    # Recherche premier mot (nettoyé de la ponctuation)
    first_word = generated_text_lower.split()[0].strip(".,!:")
    if first_word in labels:
        return first_word

    return "unknown"

"""## Fine tuning"""

"""## Error analysis
Study the errors made by the LLMs (is it more on when there are no sentences and they just give you the sentence)

# Task
Fine-tune the `google/flan-t5-small` model using QLoRA with the WordNet dataset for term type classification.

## Install necessary libraries

### Subtask:
Install the `peft` and `trl` libraries required for QLoRA fine-tuning.

**Reasoning**:
Install the required libraries `peft` and `trl` using `pip install`.
"""

"""## Load and prepare training data

### Subtask:
Load the WordNet training dataset and format it into prompt-completion pairs suitable for SFTTrainer.

**Reasoning**:
The subtask requires loading the WordNet training data and formatting it into prompt-completion pairs suitable for SFTTrainer. The first step is to load the training data using the provided `WN_TaskA_TextClf_dataset_builder` function.
"""

"""
# 1. Load the WordNet training data
id2label, label2id, train_terms, train_labels, train_sentences = WN_TaskA_TextClf_dataset_builder("train")

# 2. Create a list of available labels
available_labels = list(id2label.values())

# 3. Initialize an empty list for formatted training data
formatted_training_data = []

# 4. Iterate through the training examples
for i in range(len(train_terms)):
    term = train_terms[i]
    sentence = train_sentences[i]
    actual_label_id = train_labels[i]
    actual_label_text = id2label[actual_label_id]

    # 5. Construct the prompt string based on sentence availability
    if sentence != "":
        prompt = f"Given the term '{term}' in the sentence '{sentence}', what is the type of the term? Choose the term type from: {', '.join(available_labels)}. Answer by only giving the term type and not explaining."
    else:
        prompt = f"What is the type of the term: '{term}'? Choose from: {', '.join(available_labels)}. Answer by only giving the term type and not explaining."

    # 7. Concatenate prompt with the actual label text
    full_training_example_string = f"{prompt} {actual_label_text}"

    # 8. Append to the formatted_training_data list
    formatted_training_data.append({'text': full_training_example_string})

# 9. Convert the formatted_training_data list into a Dataset object
train_dataset = Dataset.from_list(formatted_training_data)

print(f"Created training dataset with {len(train_dataset)} examples.")
print(f"First example: {train_dataset[0]['text']}")

"""## Configure QLoRA

"""
# 1. Prepare the model for k-bit training
llm_google.gradient_checkpointing_enable()
lora_model = prepare_model_for_kbit_training(llm_google)

# 2. Define the LoRA configuration
# Inspect the model architecture to find appropriate target_modules
# For Flan-T5, common target modules include 'q', 'k', 'v' for attention layers and 'wi_0', 'wi_1' for feed-forward networks.
# Let's start with 'q', 'v' as suggested and common for T5 attention.
target_modules = ["q", "v"]

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    target_modules=target_modules
)

# 3. Apply the LoRA configuration to the model
lora_model = get_peft_model(lora_model, lora_config)

# Print trainable parameters
lora_model.print_trainable_parameters()

print("QLoRA configuration applied successfully to llm_google.")

"""

"""
# 1. Define output directory
output_dir = "./results"

# 2. Instantiate TrainingArguments with num_train_epochs set to 1
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1, # Modified to 1 epoch
    logging_steps=50,
    save_steps=200,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    report_to="none",
    remove_unused_columns=False
)

print("TrainingArguments updated successfully with num_train_epochs=1.")

# Instantiate SFTTrainer with updated training_arguments
trainer = SFTTrainer(
    model=lora_model,
    train_dataset=train_dataset,
    args=training_arguments
)

print("SFTTrainer re-instantiated successfully with updated training arguments.")

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed. Saving model...")

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")
print("Model saved successfully to './fine_tuned_model'.")

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed. Saving model...")

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")
print("Model saved successfully to './fine_tuned_model'.")
"""

def analyze_retrieval_quality(test_sentences, test_terms, train_embeddings, embedder, k_max=10):
    """
    Analyse la chute de pertinence (similarité cosine) à mesure qu'on augmente k.
    """
    print(f"--- Analyse de la qualité du voisinage (RAG) sur {len(test_sentences)} exemples ---")
    
    # 1. On prépare les queries (comme dans votre code)
    queries = []
    for s, t in tqdm(zip(test_sentences, test_terms)):
        queries.append(str(s) if len(str(s)) > 3 else str(t))
    
    # 2. Encodage des queries (Test)
    # Note: Si trop lourd, réduire à queries[:100] pour tester
    query_embeddings = embedder.encode(queries, convert_to_tensor=True).cpu().numpy()
    
    # 3. Calcul de similarité (Matrice Test x Train)
    similarities = cosine_similarity(query_embeddings, train_embeddings)
    
    # 4. On récupère les scores des k_max meilleurs voisins pour chaque phrase de test
    # On trie chaque ligne et on prend les k_max derniers (les plus grands), puis on inverse pour avoir décroissant
    # result shape: (n_test, k_max)
    top_k_values = np.sort(similarities, axis=1)[:, -k_max:][:, ::-1]
    
    # 5. Calculs statistiques
    # Moyenne de similarité pour le voisin n°1, n°2, ..., n°k
    avg_scores_per_rank = np.mean(top_k_values, axis=0)
    
    # Combien de voisins ont une similarité > 0.7 (arbitraire) en moyenne ?
    high_quality_threshold = 0.7
    avg_good_neighbors = np.mean(np.sum(top_k_values > high_quality_threshold, axis=1))

    # --- Affichage des résultats ---
    print(f"\nNombre moyen de voisins 'très pertinents' (sim > {high_quality_threshold}) : {avg_good_neighbors:.2f}")
    
    print("\nÉvolution de la similarité moyenne par rang (Rank 1 = meilleur voisin) :")
    print(f"{'Rang':<5} | {'Sim Moyenne':<15} | {'Commentaire'}")
    print("-" * 40)
    
    for i, score in enumerate(avg_scores_per_rank):
        rank = i + 1
        comment = ""
        if score > 0.85: comment = "Excellent (Copie presque conforme)"
        elif score > 0.70: comment = "Bon (Même contexte)"
        elif score > 0.50: comment = "Moyen (Thème vague)"
        else: comment = "Bruit (Peu de lien sémantique)"
        
        print(f"{rank:<5} | {score:.4f}          | {comment}")
        
    print(f"Average scores per rank: {avg_scores_per_rank}")
    return avg_scores_per_rank

def analyze_similarity_distribution(test_sentences, test_terms, train_embeddings, embedder, k_max=10):
    print(f"\n{'='*20} ANALYSE APPROFONDIE DE LA DISTRIBUTION (RAG) {'='*20}")
    
    # 1. Encodage (limité à 500 exemples pour aller vite si besoin)
    n_samples = min(len(test_sentences), 500)
    print(f"Analyse sur un échantillon de {n_samples} phrases de test...")
    
    subset_sentences = test_sentences[:n_samples]
    subset_terms = test_terms[:n_samples]
    
    queries = []
    for s, t in zip(subset_sentences, subset_terms):
        queries.append(str(s) if len(str(s)) > 3 else str(t))
        
    query_embeddings = embedder.encode(queries, convert_to_tensor=True).cpu().numpy()
    
    # 2. Calcul similarités
    similarities = cosine_similarity(query_embeddings, train_embeddings)
    
    # 3. Récupération des top-k scores pour chaque requête
    # Sort et prend les k derniers (les plus grands), puis flip pour avoir décroissant (1er = meilleur)
    top_k_values = np.sort(similarities, axis=1)[:, -k_max:][:, ::-1]
    
    # 4. Affichage du tableau de distribution
    print("\nDISTRIBUTION DES SCORES DE SIMILARITÉ PAR RANG DE VOISIN")
    print("Ce tableau montre la qualité des exemples récupérés.")
    print("-" * 105)
    print(f"{'Rang':<5} | {'Moyenne':<7} | {'> 0.90 (Exact)':<15} | {'> 0.80 (Très Bon)':<15} | {'> 0.70 (Bon)':<15} | {'> 0.60 (Moyen)':<15} | {'< 0.60 (Bruit)':<15}")
    print("-" * 105)
    
    for k in range(k_max):
        scores_at_k = top_k_values[:, k]
        avg = np.mean(scores_at_k)
        
        # Calcul des pourcentages
        p_90 = np.mean(scores_at_k >= 0.90) * 100
        p_80 = np.mean((scores_at_k >= 0.80) & (scores_at_k < 0.90)) * 100
        p_70 = np.mean((scores_at_k >= 0.70) & (scores_at_k < 0.80)) * 100
        p_60 = np.mean((scores_at_k >= 0.60) & (scores_at_k < 0.70)) * 100
        p_low = np.mean(scores_at_k < 0.60) * 100
        
        # Barre visuelle simple pour le score moyen
        bar = "#" * int(avg * 10)
        
        print(f"#{k+1:<4} | {avg:.3f}   | {p_90:5.1f}%          | {p_80:5.1f}%          | {p_70:5.1f}%          | {p_60:5.1f}%          | {p_low:5.1f}%")

    print("-" * 105)
    print("INTERPRÉTATION :")
    print(" - Si la colonne '< 0.60 (Bruit)' dépasse 50% dès le rang 3, augmenter k n'ajoute que du bruit.")
    print(" - Le modèle T5-Small est très sensible au bruit : il lui faut idéalement du '> 0.70'.\n")

if __name__ == "__main__":
    for i in range(2,11):
        print(f"Running classification for k={i}...")
        run_classification("classify_term_type_with_dynamic_few_shot", k=i)
    """
    # Charger l'embedder une seule fois
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Charger les embeddings du train (comme vous faites déjà)
    train_id2label, train_label2id, train_terms, train_labels, train_sentences = WN_TaskA_TextClf_dataset_builder("train")
    print("Encoding Train...")
    train_embeddings = embedder.encode(train_sentences, convert_to_tensor=True).cpu().numpy()

    # --- LANCER L'ANALYSE ---
    # On teste sur un sous-ensemble (ex: les 200 premiers) pour que ça aille vite
    analyze_retrieval_quality(sentences, text, train_embeddings, embedder, k_max=10)
    analyze_similarity_distribution(sentences, text, train_embeddings, embedder, k_max=10)
    """