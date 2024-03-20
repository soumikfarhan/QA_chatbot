# RAG based QA chatbot using LLM

A chatbot is a computer program that simulates human conversation. It can be used to answer questions, provide customer service, or automate tasks.

Chatbots are typically designed to mimic human conversations. They use natural language processing (NLP) to understand what people are saying and respond accordingly. Chatbots can be used to answer questions, provide customer service, or automate tasks.

I will use a Retrieval-Augmented Generation (RAG) based chatbot to answer questions, utilizing the power of the LLM api and limited training data. In my case, I will be using the chatbot to answer data about NASH. Nonalcoholic steatohepatitis, or NASH, is the most severe form of nonalcoholic fatty liver disease (NAFLD), a condition in which the liver builds up excessive fat deposits

## Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation (RAG) is used to improve the accuracy of generative AI models by providing the required custom data. In simple words, it allows the LLM’s to chat with our local or domain-specific data easily. Although LLMs like GPTs are trained on huge data, they may not have access to all the information, for example, specific domain information or confidential data within a company. RAG comes in handy in such cases as we can give the external data and retrieve the relevant information. Additionally, RAG is a cost-effective approach to maintain the accuracy and relevance of the LLM’s output.

The overview of the system can be found in the following figure:
![Overview of the System](figures/overview.png)