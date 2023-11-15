
import re
from typing import Tuple, List, Optional
from pydantic import BaseModel, RootModel, Field

import nltk
def preprocess_text(text, flag_stemm=False, flag_lemma=True, stopwords=None):
    '''
    Preprocess a string.

    Example of stopwords
    list_stopwords = nltk.corpus.stopwords.words("english")

    :parameter
        :param text: string - name of column containing text
        :param stopwords: list - list of stopwords to remove
        :param flag_stemm: bool - whether stemming is to be applied
        :param flag_lemma: bool - whether lemmitisation is to be applied
    :return
        cleaned text
    '''
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    ## substitute all characters which are NOT "word character" or "whitespace" with ""
    
    # This should be replaced by r"[^\w\s$]|(\$)[^\d]" to still capture "$200" 
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()    
    ## remove Stopwords
    if stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flag_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flag_lemma == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

class Match(BaseModel):
    span: Tuple[int,int]
    match: str

class Matches(RootModel):
    root: Optional[List[Match]]=Field(default=[])

    def __len__(self):
        return len(self.root)

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

class Tags(BaseModel):
    regex: str
    matches: Matches

class Message(BaseModel):
    """
    [
{
   "message": string
   "annotations": {
      "tags": {
         "regex": <reg-ex-pattern>
         "list": ["span": tuple(int), "match": string]
        },
      "urls": {
         "regex": <reg-ex-pattern>
         "list": ["span": tuple(int), "match": string]
        }	
    }
}
]
    
    """


    text: str
    annotations: Optional[List[Tags]] =Field(default=[])




def message_span(message:str, span:Tuple[int,int], offset:int=0):
    """
    Simply extracts the content of `message` with in indices `span`
    
    Example:
    
    message_span("Hey there, what is your name?", span=(11,15))
    
    >"what"
    
    """
    return message[(span[0]+offset):span[1]]

def find_tags(message, regex_pattern=r"(\[.?[^(\[\])]+\])"):
    """
    Default pattern for tag_pattern = r"(\[.?[^(\[\])]+\])"
    For finding urls, use tag_pattern = r"(?:(?:https?|ftps?):\/\/)[\w/\-?=%.]+\.[\w/\-&?=%.]+"
    
    """

    tag_list = []
    for item in re.finditer(regex_pattern, message):
        span = item.span(0)
        match = message_span(message, span)
        tag_list.append({"span":span, "match":match})
    return tag_list



def create_message_entry(message, tag_pattern=r"(\[.?[^(\[\])]+\])", url_pattern=r'(?:(?:https?|ftps?):\/\/)[\w/\-?=%.]+\.[\w/\-&?=%.]+'):
    """
    tag_pattern = r"(\[.?[^(\[\])]+\])"
    """

    entry = {"message":message, "annotations":
             {"tags": {"regex":tag_pattern, 
                       "list":[]}
             , 
              "urls":{"regex": url_pattern, 
                      "list":[]}
             }
            }
    entry["annotations"]["tags"]["list"] = find_tags(message=message, regex_pattern=tag_pattern)
    entry["annotations"]["urls"]["list"] = find_tags(message=message, regex_pattern=url_pattern)
    
    return entry



def end_match_lookup(start_match):
    """
    This implements a lookup table for each start tag, e.g `[quote...]` we know 
    the tag that ends the section, e.g. `[/quote]` in this case
    
    """
    if start_match.startswith("[/"):
        return None, None
    elif start_match.startswith("[quote"):
        return ("quote","quote", "[/quote]")
    elif start_match.startswith("[user"):
        return ("user", "user","[/user]")
    elif start_match.startswith("[i]"):
        return ("italic", "i", "[/i]")
    elif start_match.startswith("[url"):
        return ("url", "url", "[/url]")
    elif start_match.startswith("[URL="):
        return ("URL", "URL=.+", "[/URL]")
    elif start_match.startswith("[b]"):
        return ("bold", "b", "[/b]")
    
def search_close_tag(tag_list):
    """
    tag_list is a list with dictionary entries with keys "match" and "scan"
    It will start going through the list, from the first start tag, to the closening tag at the 
    same depth and of the same type. 
    
    That match will be returned as a dictionary with keys 
    * `"type"`, 
    * `"span"` 
    * `"success"` (True if closing tag was found)
    
    Example
    First output:
        {'type': 'quote', 'span': (42, 400), 'success': True}
    Second output, the remaning list from the input `tag_list`:
    
     [
      {'span': (415, 418), 'match': '[i]'},
      {'span': (421, 425), 'match': '[/i]'},
      {'span': (468, 474), 'match': '[user]'},
      {'span': (481, 488), 'match': '[/user]'}
      ]
      
    
    If no closing tag was found, in this case we deliver span from opening until the last 
    span found in the `tag_list`. This is probably not the proper span, which should be the 
    end of the message, rather than the last found tag, but this function only knows about the `tag_list`
    
    """
    
    # prune away entries from the list which are closing tags
    while tag_list[0]["match"].startswith("[/"):
        tag_list = tag_list[1:]
        
    
    start_tag = tag_list[0]
    start_span = start_tag["span"][0]
    match_type, match_string, end_match = end_match_lookup(start_tag["match"])
    
    # Keep track of the depth from the opening tag
    depth = 0
    for idx, tag in enumerate(tag_list[1:]):
        
        if depth==0 and tag["match"] == end_match:
            final = {"type":match_type, "match_string": match_string, "span":(start_span, tag["span"][1]), "success":True}
            rest_list = tag_list[(idx+2):]
            
            return final, rest_list
        
        if tag["match"].startswith("[/"):
            depth -= 1
        else:
            depth += 1
    
    # If we end up here, no closing tag was found, in this case we deliver span from opening until the last 
    # span found in the `tag_list`. This is probably not the proper span, which should be the end of the message, 
    # rather 
    
    return {"type":match_type,"span":(start_span, tag["span"][1]), "success":False}, []

def check_balanced_tag_list(tag_list):
    """
    Check if the tag_list is balanced, i.e. if there are equal number of start tags 
    and closing tags
    
    Returns true if there are equally many start and closing tags
    """
    
    depth = 0
    for item in tag_list:
        if not item["match"].startswith("[/"):
            depth += 1
        else:
            depth -= 1
    
    return depth == 0
    

def process_tags(tag_list, message=None):
    """
    This extracts the paired tags at depth 0 from the tag_list
    
    
    """
    
    result = []
    work_list = tag_list
    while len(work_list)>1:
        
        
        # note that the returning `work_list` will be left-truncated after returning from `search_close_tag`
        res, work_list = search_close_tag(work_list)

        if message is not None:
            cut_message = message_span(message, res["span"])
            res.update({"snippet":cut_message})

        result.append(res)

          
    return result
        

# Default rule is to extract the content inside the tag-marks, so if not in the dictionary, 

replace_rules = {"quote": "<quote-of-somebody>"
                 , "url":"<url-to-something>"
                 , "URL":"<url-to-something>"
                 , 'raw_url': "<url-to-something>" }     # <- this is a URL found without the tag-brackets




def replace_in_string(message, tag:dict, replace_rules=replace_rules):
    """
    This is a sub-function to "process_string" and operates on the original full "message"
    and each found tag at a time. The 
    
    "tag" is a dictionary with fields "type". If the replace_rule for this particular type is NOne, 
    then also "match_string" and "span" is required.

    replace_rules holds the replace strings for each "type" 
    If there is no replace string for a specific type, the content "inside content" of the tag 
    is returned, e.g. "hej" if the input if message="<i>hej</i>"
    
    """
    replace_string = replace_rules.get(tag["type"])

    if replace_string is None:
        tag_string = tag.get("match_string")
        tag_span = tag["span"]
        snippet = message_span(message, tag_span)
        snippet = re.findall(f"\[{tag_string}\](.+)\[/{tag_string}\]", snippet)[0]
    else:
        snippet = replace_string    
    
    return snippet
    
def process_string(message, matched_tags, replace_rules=replace_rules):
    """
    "message" is the original string. 
    
    "matched_tags" is a list of dicts with required keys 
       * "type": type of tag "url", "quote", "user", "italic", "bold"
       * "span":  tuple of indices in "message" 
     and optional keys 
       * "success": Boolean 
       * "snippet": part of "message" corressponding to the "span")
    
    Return:
    * output: string of the filtered "message" according to rules
    
    """
    
    start_idx = 0
    output = ""
    for tag in matched_tags:
        tag_span = tag["span"]
        output = output +  message[start_idx: tag_span[0]]
        output = output + replace_in_string(message, tag, replace_rules=replace_rules)
        start_idx = tag_span[1]
        
    output = output + message[start_idx:]
    return output


def remove_urls(message, replace_string="", regex_pattern=r"(?:(?:https?|ftps?):\/\/)[\w/\-?=%.]+\.[\w/\-&?=%.]+"):
    """
    Main function for for removing stand-alone URLs (that is, NOT inside tags-brackets like [url] and [/url]) in a message


    Default, regex_pattern to detect URLs is
    r"(?:(?:https?|ftps?):\/\/)[\w/\-?=%.]+\.[\w/\-&?=%.]+"

    The occurrence of the URL will be replaced by what is in "replace_string" (default is empty string "")

    Example ( default behaviour):

    remove_urls("here is a URL http://www.google.se")
    > 'here is a URL '

    Example (with specified replace_string):

    remove_urls("here is a URL http://www.google.se", replace_string="<URL>")
    > 'here is a URL <URL>'


    """


    tag_list = find_tags(message, regex_pattern)

    if replace_string is None:
        raise ValueError("'replace_string' must be a string, cannot be None. None is used for proper tags only" )

    replace_rules = {"raw_url": replace_string}
    
    tag_list = [{**item, "type":"raw_url"} for item in tag_list]

    return process_string(message, tag_list, replace_rules)

