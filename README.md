# Information Retrieval Project
Our code is divided into frontend and backend.
All the logic is written in the backend, and the call to the appropriate functions is done from the frontend.

The functions available in the frontend are the functions given to us in advance:
- Search
- Search_body
- Search_title
- Search_anchor
- Set_pagerank
- Set_pageview

The classes available in the backend are the classes given to us in Assignment 3 and 4 which we have tempered with to intergrate it to our system:
- MultiFileWriter - Sequential binary writer to multiple files of up to BLOCK_SIZE each
- MultiFileReader - Sequential binary reader of multiple files of up to BLOCK_SIZE each
- InvertedIndex - Initializes the inverted index and add documents to it (if provided)

The functions available in the backend are the functions:
- Read_pkl - external function used to read pickle files (like the index file)
- Tokenize - The same tokenizer from assignment 3
- Read_posting_list - returnes the posting list of specific word from specific index
- Autocorrect - correct typo and spelling mistakes. gets tokenized query based on the client input, and the relevant index
- Cossine_sim - fast cossine similarity method, which avoids calculating all the calculations that can be done offline
- Precision_at_k - evaluation method
- Precision_and_recall - another evaluation method. returns a tuple of (precision,recall) 


Code Stroage:
  - All of the pickles and the csv file for the global variables should be in the working directory. 
  - The bin files and the other files of the body index - bdf and bpl need to be in a sub directory of the working directory by the name of "body_index"  
  - The bin files and the other files of the title index - tdf and tpl need to be in a sub directory of the working directory by the name of "title_index"
  - All the indexes,  the pickles we created and the page rank are in the bucket to which a link is attached in the project report.
  - They were not additionally uploaded here due to the file upload size limit of the git.
