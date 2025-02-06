# English prompt

sft_prompt_en = "{instruction} {response}"

all_prompt_en = {}

# =====================================================
# Task -- Item2Index
# =====================================================

item2index_prompt = []

#####——1
prompt = {}
prompt[
    "instruction"] = "Given the description of an item {content}, predict the corresponding item."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——2
prompt = {}
prompt[
    "instruction"] = "Based on the item description {content}, predict the related item."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——3
prompt = {}
prompt[
    "instruction"] = "Using the provided description {content}, predict the corresponding item."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——4
prompt = {}
prompt[
    "instruction"] = "Given the content description of an item {content}, forecast the matching item."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——5
prompt = {}
prompt[
    "instruction"] = "Predict the relevant item based on the provided description {content}."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——6
prompt = {}
prompt[
    "instruction"] = "Given the item content {content}, predict the item that corresponds to it."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——7
prompt = {}
prompt[
    "instruction"] = "Using the item description {content}, identify the corresponding item."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——8
prompt = {}
prompt[
    "instruction"] = "Based on the item content {content}, predict the associated item."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——9
prompt = {}
prompt[
    "instruction"] = "Given the description of the item {content}, determine the corresponding item."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——10
prompt = {}
prompt[
    "instruction"] = "Using the item’s description {content}, predict the relevant item."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

all_prompt_en["item2index"] = item2index_prompt

# =====================================================
# Task -- Index2Item
# =====================================================

index2item_prompt = []

#####——1
prompt = {}
prompt[
    "instruction"] = "Given the item {item}, please provide its description."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "Based on the item {item}, give a description of it."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Please provide a description for the item {item}."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "Given the item {item}, describe it."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——5
prompt = {}
prompt[
    "instruction"] = "Using the item {item}, please generate its description."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——6
prompt = {}
prompt["instruction"] = "Please provide the description of the item {item}."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "From the item {item}, please write its description."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "Please give a description of the item {item}."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——9
prompt = {}
prompt[
    "instruction"] = "Based on {item}, provide the corresponding description."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "Using the item {item}, please produce a description."
prompt["response"] = "{content}"
index2item_prompt.append(prompt)

all_prompt_en["index2item"] = index2item_prompt

# ========================================================
# Task -- Sequential Recommendation
# ========================================================

seqrec_prompt = []

#####——1
prompt = {}
prompt[
    "instruction"] = "Here is the user's history {history}. Please recommend the next item the user might want to click on."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——2
prompt = {}
prompt[
    "instruction"] = "The following is the user's historical behavior {history}. Please suggest the next item the user may be interested in clicking."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——3
prompt = {}
prompt[
    "instruction"] = "Below is the user's history {history}. Please recommend the next item the user is likely to click on."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——4
prompt = {}
prompt[
    "instruction"] = "Here is the user's past behavior {history}. Based on this, please recommend the next item they might click."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——5
prompt = {}
prompt[
    "instruction"] = "The user's history is as follows {history}. Please predict and recommend the next item the user may want to click."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——6
prompt = {}
prompt[
    "instruction"] = "Below are the user's previous behaviors {history}. Please suggest the next item the user might want to interact with."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——7
prompt = {}
prompt[
    "instruction"] = "The user's historical behaviors are listed below {history}. Please recommend the next item the user is likely to be interested in clicking."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——8
prompt = {}
prompt[
    "instruction"] = "Here is a record of the user's history {history}. Based on this, please recommend the next item for the user to click on."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——9
prompt = {}
prompt[
    "instruction"] = "The following is the user's behavior history {history}. Please recommend the next item the user might want to engage with."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

#####——10
prompt = {}
prompt[
    "instruction"] = "Below is the user's historical data {history}. Please suggest the next item the user could be interested in clicking."
prompt["response"] = "{target}"
seqrec_prompt.append(prompt)

all_prompt_en["seqrec"] = seqrec_prompt

# ========================================================
# Task -- Query Recommendation
# ========================================================

queryrec_prompt = []

#####——1
prompt = {}
prompt[
    "instruction"] = "Here is the user's history {history}. Please predict the next query the user might want to search."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——2
prompt = {}
prompt[
    "instruction"] = "The following is the user's historical data {history}. Please predict the user's next potential search query."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——3
prompt = {}
prompt[
    "instruction"] = "Below is the user's history {history}. Please forecast the next query the user is likely to search."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——4
prompt = {}
prompt[
    "instruction"] = "Here is a record of the user's past behaviors {history}. Please predict the next search query the user may want to input."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——5
prompt = {}
prompt[
    "instruction"] = "The user's historical behavior is provided below {history}. Please predict what the user might search for next."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——6
prompt = {}
prompt[
    "instruction"] = "Below are the user's previous behaviors {history}. Please predict the user's next intended search query."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——7
prompt = {}
prompt[
    "instruction"] = "The following shows the user's history {history}. Based on this, please predict the next query the user is likely to search."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——8
prompt = {}
prompt[
    "instruction"] = "Here is the user's behavioral history {history}. Please predict the next search query they might be interested in."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——9
prompt = {}
prompt[
    "instruction"] = "The user's past behaviors are as follows {history}. Please forecast what the user might want to search for next."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

#####——10
prompt = {}
prompt[
    "instruction"] = "Below is a summary of the user's history {history}. Please predict the next search query the user may input."
prompt["response"] = "{target}"
queryrec_prompt.append(prompt)

all_prompt_en["queryrec"] = queryrec_prompt

# ========================================================
# Task -- Personalized Search
# ========================================================

persrc_prompt = []

#####——1
prompt = {}
prompt[
    "instruction"] = "{history} The following is the user's search query {query}. Please predict the next item the user will click on after the search."
prompt["history"] = "Here is the user's history {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——2
prompt = {}
prompt[
    "instruction"] = "{history} Along with the search query {query}. Please forecast the next item the user is likely to click on."
prompt["history"] = "Below is the user's historical data {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——3
prompt = {}
prompt[
    "instruction"] = "{history} The user's search query is {query}. Please predict the next item they will click after the search."
prompt["history"] = "Here is a record of the user's behaviors {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——4
prompt = {}
prompt[
    "instruction"] = "{history} Along with their search query {query}. Please predict the next item the user might click."
prompt["history"] = "The user's past behavior is shown below {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——5
prompt = {}
prompt[
    "instruction"] = "{history} And their recent search query {query}. Please predict the next item the user is expected to click."
prompt["history"] = "Below are the user's historical behaviors {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——6
prompt = {}
prompt[
    "instruction"] = "{history} And the search query they entered {query}. Please predict the next item the user will likely click on."
prompt["history"] = "Here is the user's behavior history {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——7
prompt = {}
prompt[
    "instruction"] = "{history} And their current search query {query}. Please forecast the next item they will click on."
prompt["history"] = "The following shows the user's history {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——8
prompt = {}
prompt[
    "instruction"] = "{history} Their search query is {query}. Please predict the next item they might click following the search."
prompt["history"] = "Here is the user's behavioral record {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——9
prompt = {}
prompt[
    "instruction"] = "{history} And the search query {query}. Please predict which item the user will click on next."
prompt["history"] = "Below is a summary of the user's history {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

#####——10
prompt = {}
prompt[
    "instruction"] = "{history} And their search query {query} are provided. Please forecast the next item the user will click on."
prompt["history"] = "The user's historical data {history}."
prompt["response"] = "{target}"
persrc_prompt.append(prompt)

all_prompt_en["persrc"] = persrc_prompt
