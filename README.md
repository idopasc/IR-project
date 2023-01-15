# IR-project
Our code is divided into frontend and backend.
All the logic is written in the backend, and the call to the appropriate functions is done from the frontend.

The functions available in the frontend are the functions given to us in advance:
- search
- search_body
- search_title
- search_anchor
- get_pagerank
- get_pageview

The classes available in the backend are the classes given to us in Assignment 3 and 4 which we have tempered with to intergrate it to our system:
- MultiFileWriter - Sequential binary writer to multiple files of up to BLOCK_SIZE each
- MultiFileReader - Sequential binary reader of multiple files of up to BLOCK_SIZE each
- InvertedIndex - Initializes the inverted index and add documents to it (if provided)

The functions available in the backend are the functions:
- read_pkl - external function used to read pickle files (like the index file)
- tokenize - The same tokenizer from assignment 3
- read_posting_list - returnes the posting list of specific word from specific index
- autocorrect - correct typo and spelling mistakes. gets tokenized query based on the client input, and the relevant index
- cossine_sim - fast cossine similarity method, which avoids calculating all the calculations that can be done offline
