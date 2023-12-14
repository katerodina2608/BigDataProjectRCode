library(tidyverse)
library(tidytext)
library(topicmodels)
library(caret)
library(mlbench)
library(tm)
library(quanteda)
library(textdata)
library(stringr)
library(kableExtra)
library(caret)
library(mlbench)
library(LogicReg)
library(ggmosaic)

data("stop_words")

#bindings two datasets by rows
chatgpt <- chatgpt %>%
  select(text, id)
midjourney <- midjourney %>%
  select(text, id)
genai <- rbind(chatgpt, midjourney)


#data cleaning

#creating a corpus from the dataframe to use tm_map() function
names(genai)[names(genai)== "id"] <- "doc_id"
names(genai)[names(genai)== "body"] <- "text"
genai_corpus <- Corpus(DataframeSource(genai))
#lowering case
genai_corpus <- tm_map(genai_corpus, tolower)

#removing punctuation
genai_corpus<- tm_map(genai_corpus, removePunctuation)

#removing numbers
genai_corpus<- tm_map(genai_corpus, removeNumbers)

#removing special characters, http links, and words with 2 or less characters
text_clean <- function(x) {
  gsub('http\\S=\\s*','', x)
  gsub('[[:cntrl:]]','',x)
  gsub('\\b\\w{1,2}\\b','',x)
}
genai_corpus <-tm_map(genai_corpus ,text_clean)

#removing stopwords + custom words (chatgpt and gpt removed due to very high occurence)
genai_corpus <- tm_map(genai_corpus, removeWords, c(stopwords('english'),"chatgpt","midjourney","gpt","artificialintelligence"))


#converting a corpus back to a dataframe
genai_clean <- data.frame(text = get("content", genai_corpus))
genai_clean <- genai %>%
  mutate(
    text = genai_clean$text,
    rawtext = genai$text
  )
genai_clean <- genai_clean[ , ! names(genai_clean) %in% c("rawtext")]



#tokenizing the documents

count_genai <- genai_clean %>%
  unnest_tokens(word, text) %>%
  count(doc_id, word, sort = TRUE) %>%
  anti_join(stop_words)

#selecting words that occur in at least 10 documents
words_10 <- count_genai %>%
  group_by(word) %>%
  summarise(n = n()) %>% 
  filter(n >= 10) %>%
  select(word)

#creating a document-term matrix where each unique word is represented in a column and each document is a row. 
#Entries are the counts of words in each document
#to decrease the sparsity we only consider words that occur in at least 10 documents
genai_dtm <- count_genai %>% 
  right_join(words_10, by = "word") %>%
  count(doc_id, word, n) %>%
  cast_dtm(doc_id, word, n)


#creating an LDA model with 2 topics
genai_lda2 <- LDA(genai_dtm, k = 2, control=list(seed=101))

genai_topics <- tidy(genai_lda2, matrix="beta")
genai_top_terms <- genai_topics %>% group_by(topic) %>%top_n(10, beta)%>%ungroup()%>%arrange(topic, -beta)
topic_graph <- genai_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term,beta,fill=factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~topic, scales="free")+
  coord_flip()
topic_graph

#creating a dataframe of comments which include top 10 words for each topic
#using anti_join function to avoid overlap between posts on chatgpt and midjourney

chatgpt_wordlist <- c("openai", "check", "monkai", "amp", "crypto", "nft", "bot", "technology", "join", "free")
midjourney_wordlist <- c("aiart", "art", "aiartcommunity", "digitalart", "aiartwork", "aiartists", "nftart", "artist", "machinelearning", "scifi")

chatgpt_word_data <- genai_clean %>%
  filter(str_detect(genai_clean$text, paste(chatgpt_wordlist, collapse = "|"))) %>%
  anti_join(genai_clean %>%
             filter(str_detect(genai_clean$text, paste(midjourney_wordlist, collapse = "|"))))

midjourney_word_data <- genai_clean %>%
  filter(str_detect(genai_clean$text, paste(midjourney_wordlist, collapse = "|"))) %>%
  anti_join(chatgpt_word_data)


# tokenize the documents for each topic 
count_chatgpt_sent <- chatgpt_word_data %>%
  unnest_tokens(word, text) 

count_midjourney_sent <- midjourney_word_data %>%
  unnest_tokens(word, text)


# run the sentiment analysis on tokenized documents for each topic
get_sentiments("bing")
# get the sentiments for each group


sentiment_midjourney <- count_midjourney_sent %>%
  group_by(word) %>%
  inner_join(get_sentiments("bing"))
  

sentiment_gpt <- count_chatgpt_sent %>%
  group_by(word) %>%
  inner_join(get_sentiments("bing")) %>%
  ungroup()


# creating a combined dataframe with sentiment words from ChatGPT and MidJourney wordlists
sentiment_gpt <- sentiment_gpt %>%
  mutate(
    dataset = "ChatGPT"
  )

sentiment_midjourney <- sentiment_midjourney %>%
  mutate(
    dataset = "MidJourney"
  )
sentiment_genai <- rbind(sentiment_gpt, sentiment_midjourney)
sentiment_genai <- sentiment_genai[,-1]
sentiment_genai$sentiment <- as.factor(sentiment_genai$sentiment)


# Results: descriptive statistics and visualizations

descr_stats <- sentiment_genai %>%
  group_by(dataset, sentiment) %>%
  summarize(
    n = n()) %>%
  mutate(
    Proportion = round(n/sum(n),2)
  )
descr_stats


#Logistic regression model

#labeling levels for the sentiment" variable. 0 - Negative; 1 - Positive
sentiment_genai <- sentiment_genai %>% 
  mutate(
  sentiment = factor(sentiment,
                     levels = c("negative", "positive"),
                     labels = c("0", "1"))
)
is.factor(sentiment_genai$sentiment)



#create training and testing datasets
# 75% of the sample size
set.seed(11)
smp_size <- floor(0.75 * nrow(sentiment_genai))

train_data <- sample(seq_len(nrow(sentiment_genai)), size = smp_size)

train_genai <- sentiment_genai[train_data, ]
test_genai <- sentiment_genai[-train_data, ]


# Cross-validation,10 K-folds
train.control <- trainControl(method = "cv", number = 10)

#run the null model on train dataset

m1 <- train(sentiment ~ dataset, data = sentiment_genai, method = "glm", family = "binomial", trControl = train.control)
matrix <- confusionMatrix.train(m1)
matrix

# applying the model to test data
m1pred <- predict(m1, newdata = test_genai, type = "prob")
m1pred <- cbind(m1pred, test_genai)

m1pred2<- m1pred %>%
  mutate(correct = if_else(positive > .5 & sentiment == 1, "Correct", "Incorrect"))%>%
  count(correct)
m1pred2
