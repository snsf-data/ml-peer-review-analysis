# ------------------------------------------------------------------------------------------------------------------- #
# Script to define prompts for Meta's Llama3 8B Instruct model for all categories                                     #
# ------------------------------------------------------------------------------------------------------------------- #

# define prompts for all 12 categories
# chat template with few-shot learning (https://huggingface.co/docs/transformers/main/chat_templating)

# ------------------------------------------------------------------------------------------------------------------- #
# criterion_track_record
criterion_track_record = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Track Record and Expertise' and it refers to the following question: 'Are the scientific qualifications of each applicant or team addressed?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'The applicants have considerable expertise in these approaches and present preliminary experimental data which support the proposal.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# criterion_relevance_originality_topicality
criterion_relevance_originality_topicality = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Relevance, Originality, and Topicality' and it refers to the following question: 'Is the proposed research projects scientific relevance or impact or originality addressed?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'My judgment on the relevance and originality is similar as last time - I think the topic is highly relevant, original, and topical.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# criterion_suitability
criterion_suitability = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Suitability' and it refers to the following question: 'Is the suitability of the methods to be used within the proposed research project addressed?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'The meths are suitable and can be clearly achieved in this laboratory.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# criterion_feasibility
criterion_feasibility = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Feasibility' and it refers to the following question: 'Is the feasibility of the proposed research project addressed?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'The proposed project is feasible based on their expertise.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# candidate_other
candidate_other = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Applicant' and it refers to the following question: 'Is the applicant or team or their qualifications addressed?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'Therefore, they have ability to carry out the proposed project based on their strong expertise in this special research area and their outstanding track record.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# candidate_quantity
candidate_quantity = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Applicant: Quantity' and it refers to the following question: 'Is the applicant or team with regard to quantitative indicators addressed?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'He is first author of eight manuscripts and reviews and for four manuscript he is last author, and in addition he is joint last author on one additional manuscript.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# proposal_general
proposal_general = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Proposal' and it refers to the following question: 'Is the proposal and specific parts of it (except for methods) addressed as opposed to the applicant(s), or context beyond the proposal (such as the research field or the funding schemes objectives etc.)?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'The proposal not only has a strong rationale, it also incorporates original ideas and questions that will be tested using modern approaches.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# proposal_method
proposal_method = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Methods' and it refers to the following question: 'Are the methods proposed in the application addressed?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'The reduced-form methods suggested in the proposal seem to be the appropriate method for the proposed projects.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# positive
positive = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Positive Statement' and it refers to the following question: 'Is the sentence itself a positive statement or does it contain a positive statement?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'The strengths of this proposal are the strong applicant and team, the current dynamic nature of the field under study, and the potential impact of the work on advancing biological understanding and clinical care.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# negative
negative = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Negative Statement' and it refers to the following question: 'Is the sentence itself a negative statement or does it contain a negative statement?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'There are a lot of spelling mistakes and references that are misleading or difficult to understand.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# suggestion
suggestion = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Suggestion' and it refers to the following question: 'Is the sentence suggesting how to improve the proposal?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'If the project is accepted, then I would suggest that the project pays specific attention to increase practical relevance.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# rationale
rationale = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. You are allowed to answer only with one single word.",
    },
    {
        "role": "user",
        "content": "The sentences from a grant peer review can be labelled as <CATEGORY-YES> if they refer to a given category or <CATEGORY-NO> if they do not refer to the given category. The given category is called 'Rationale' and it refers to the following question: 'Is a rationale for the positive and negative statements provided?'. Please provide the label based on the given sentence in the following format: <label>"
    },
    {
        "role": "assistant",
        "content": "Could you provide some examples of sentences from the grant peer review with their corresponding label?",
    },
    {
        "role": "user",
        "content": "Yes. Here is an example for the label <CATEGORY-YES>: 'Given the comments above, this reviewer considers that the proposed methods are well suited to provide key answers to the proposed hypotheses, particularly because these methods and models have provided the background knowledge justifying the reverse engineering approach.'. And here is an example for the label <CATEGORY-NO>: 'There are currently several activities on an international level that have identified the issue and activities are underway.' ",
    },
    {
        "role": "assistant",
        "content": "Thank you for the examples. Please provide the sentence you would like me to label. I will respond with the appropriate label in the format <CATEGORY-YES> or <CATEGORY-NO>",
    },
    {
        "role": "user",
        "content": "",
    },
]

# ------------------------------------------------------------------------------------------------------------------- #
# combine all the prompts into a named dictionary
prompts = {"criterion_track_record": criterion_track_record,
           "criterion_relevance_originality_topicality": criterion_relevance_originality_topicality,
           "criterion_suitability": criterion_suitability,
           "criterion_feasibility": criterion_feasibility,
           "candidate_other": candidate_other,
           "candidate_quantity": candidate_quantity,
           "proposal_general": proposal_general,
           "proposal_method": proposal_method,
           "positive": positive,
           "negative": negative,
           "suggestion": suggestion,
           "rationale": rationale}
# ------------------------------------------------------------------------------------------------------------------- #
