# Utilidades simples para leer AnatEM y preparar datos de NER (BIO).

import os
import re
import random

def validate_sentence_alignment(tokens, tags):
    """Check basic token-tag alignment"""
    if len(tokens) != len(tags):
        raise ValueError(f"Sentence count mismatch: {len(tokens)} vs {len(tags)}")
    
    for i, (sent_tokens, sent_tags) in enumerate(zip(tokens, tags)):
        if len(sent_tokens) != len(sent_tags):
            raise ValueError(f"Sentence {i} length mismatch: {len(sent_tokens)} vs {len(sent_tags)}")

def validate_bio_format(tags, max_errors=5):
    """Check BIO tag sequence validity"""
    errors = []
    for i, sent_tags in enumerate(tags):
        for j, tag in enumerate(sent_tags):
            if tag.startswith('I-'):
                entity_type = tag[2:]
                if j == 0 or not sent_tags[j-1].endswith(entity_type):
                    errors.append(f"Invalid I- tag at sentence {i}, position {j}")
                    if len(errors) >= max_errors:
                        return errors
    return errors

def log_data_stats(tokens, tags):
    """Print basic data statistics"""
    lengths = [len(sent) for sent in tokens]
    print(f"Sentences: {len(tokens)}")
    print(f"Avg length: {sum(lengths)/len(lengths):.1f} tokens")
    print(f"Max length: {max(lengths)} tokens")

def validate_sentence_alignment(tokens, tags):
    """Valida que tokens y tags tengan la misma longitud por oración"""
    if len(tokens) != len(tags):
        raise ValueError(f"Mismatch: {len(tokens)} oraciones vs {len(tags)} secuencias de tags")
    
    for i, (sent_tokens, sent_tags) in enumerate(zip(tokens, tags)):
        if len(sent_tokens) != len(sent_tags):
            raise ValueError(f"Oración {i}: {len(sent_tokens)} tokens vs {len(sent_tags)} tags")

def validate_bio_sequences(tags, max_errors=5):
    """Valida secuencias BIO básicas"""
    errors = 0
    for i, sent_tags in enumerate(tags):
        for j, tag in enumerate(sent_tags):
            if tag.startswith('I-') and j > 0:
                entity_type = tag[2:]
                prev_tag = sent_tags[j-1]
                if not prev_tag.endswith(entity_type):
                    print(f"Warning: Secuencia BIO inválida en oración {i}, posición {j}: {tag}")
                    errors += 1
                    if errors >= max_errors:
                        return

def log_data_stats(tokens, tags):
    """Registra estadísticas básicas de los datos"""
    lengths = [len(sent) for sent in tokens]
    all_tags = [tag for sent in tags for tag in sent]
    from collections import Counter
    tag_counts = Counter(all_tags)
    
    print(f"Oraciones cargadas: {len(tokens)}")
    print(f"Longitud promedio: {sum(lengths)/len(lengths):.1f} tokens")
    print(f"Tags más comunes: {dict(list(tag_counts.most_common(5)))}")

def is_bio_tag(s):
    return bool(re.match(r"^(B|I)-", s)) or s == "O"

def read_conll_like_file(path):
    """Read CoNLL-style files with basic validation"""
    sentences_tokens = []
    sentences_tags = []
    cur_toks, cur_tags = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur_toks:
                    sentences_tokens.append(cur_toks)
                    sentences_tags.append(cur_tags)
                    cur_toks, cur_tags = [], []
                continue

            cols = line.split("\t") if "\t" in line else line.split()
            if len(cols) < 2:
                continue

            a, b = cols[0], cols[1]
            if is_bio_tag(a) and not is_bio_tag(b):
                tag, token = a, b
            elif is_bio_tag(b) and not is_bio_tag(a):
                token, tag = a, b
            else:
                token, tag = a, b
                if not is_bio_tag(tag):
                    tag = "O"

            cur_toks.append(token)
            cur_tags.append(tag)

    if cur_toks:
        sentences_tokens.append(cur_toks)
        sentences_tags.append(cur_tags)

    return sentences_tokens, sentences_tags

def read_nersuite_file(path):
    """Read NERsuite format with basic validation"""
    sentences_tokens = []
    sentences_tags = []
    cur_toks, cur_tags = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur_toks:
                    sentences_tokens.append(cur_toks)
                    sentences_tags.append(cur_tags)
                    cur_toks, cur_tags = [], []
                continue

            cols = line.split("\t")
            if len(cols) >= 4:
                tag = cols[0]
                token = cols[3]
                if not is_bio_tag(tag):
                    tag = "O"
                cur_toks.append(token)
                cur_tags.append(tag)

    if cur_toks:
        sentences_tokens.append(cur_toks)
        sentences_tags.append(cur_tags)

    return sentences_tokens, sentences_tags

def collect_files(dir_path):
    files = []
    if os.path.isdir(dir_path):
        for fn in sorted(os.listdir(dir_path)):
            p = os.path.join(dir_path, fn)
            if os.path.isfile(p):
                files.append(p)
    return files

def read_split_dir(root, fmt, split_name):
    """
    Lee root/<fmt>/<split_name> si existe. Retorna listas de oraciones (tokens, tags).
    fmt: 'conll' | 'nersuite' | 'nersuite-spanish'
    """
    split_dir = os.path.join(root, fmt, split_name)
    files = collect_files(split_dir)

    all_tokens, all_tags = [], []
    for p in files:
        if fmt == "nersuite":
            toks, tags = read_nersuite_file(p)
        else:
            toks, tags = read_conll_like_file(p)
        all_tokens.extend(toks)
        all_tags.extend(tags)

    validate_sentence_alignment(all_tokens, all_tags)

    return all_tokens, all_tags

def read_flat_dir(root, fmt):
    """
    Lee root/<fmt>/ si no hay subcarpetas train/devel/test.
    Une todas las oraciones y devuelve (tokens, tags).
    """
    base_dir = os.path.join(root, fmt)
    files = collect_files(base_dir)

    all_tokens, all_tags = [], []
    for p in files:
        if fmt == "nersuite":
            toks, tags = read_nersuite_file(p)
        else:
            toks, tags = read_conll_like_file(p)
        all_tokens.extend(toks)
        all_tags.extend(tags)
    
    validate_sentence_alignment(all_tokens, all_tags)

    return all_tokens, all_tags

def auto_split(tokens, tags, seed=42, ratios=(0.8, 0.1, 0.1)):
    """
    Divide por oración en train/dev/test manteniendo orden aleatorizado (semilla fija).
    Retorna ((train_tokens, train_tags), (dev_tokens, dev_tags), (test_tokens, test_tags)).
    """
    idxs = list(range(len(tokens)))
    random.Random(seed).shuffle(idxs)
    n = len(idxs)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    train_idx = idxs[:n_train]
    dev_idx = idxs[n_train:n_train+n_dev]
    test_idx = idxs[n_train+n_dev:]

    def gather(idxs_):
        return [tokens[i] for i in idxs_], [tags[i] for i in idxs_]

    return gather(train_idx), gather(dev_idx), gather(test_idx)

def build_label_list(train_tags, dev_tags, test_tags):
    """
    Construye lista de etiquetas BIO a partir de los tres splits.
    Devuelve: ["O", "B-...", "I-...", ...]
    """
    uniq = set()
    for seqs in (train_tags, dev_tags, test_tags):
        for sent in seqs:
            for t in sent:
                uniq.add(t)
    bio = sorted([x for x in uniq if x != "O"])
    return ["O"] + bio if uniq else ["O"]

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length):
    """
    Tokeniza con is_split_into_words=True y alinea etiquetas a sub-tokens.
    Sub-tokens extra reciben -100 para que no entren a la pérdida.
    """
    enc = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=max_length
    )

    out_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = enc.word_ids(i)
        prev = None
        lab_ids = []
        for w_idx in word_ids:
            if w_idx is None:
                lab_ids.append(-100)
            elif w_idx != prev:
                lab_ids.append(label2id[tags[w_idx]])
            else:
                lab_ids.append(-100)
            prev = w_idx
        out_labels.append(lab_ids)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": out_labels
    }
