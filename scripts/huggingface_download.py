from datasets import load_dataset

sentence_level = load_dataset("UniversalCEFR/readme_en")
sentence_level["train"].to_json("readme_train.jsonl", lines=True)

print(sentence_level)
print(sentence_level['train'][0])

document_level_cambridge = load_dataset("UniversalCEFR/cambridge_exams_en")
document_level_cambridge["train"].to_json("cambridge_train.jsonl", lines=True)

print(document_level_cambridge)
print(document_level_cambridge['train'][0])

document_level_elg = load_dataset("UniversalCEFR/elg_cefr_en")
document_level_elg["train"].to_json("elg_train.jsonl", lines=True)

print(document_level_elg)
print(document_level_elg['train'][0])
