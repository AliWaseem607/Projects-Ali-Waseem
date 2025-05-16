# additional tools that I may need that don't fit into other text files
import ctypes
# takes in a url and a set of href links, returns a set of links that can be
# followed to access the xml files to be downloaded
def create_urls(web_url,link_set):
    full_links = set()
    for link in link_set:
        x = web_url + str(link)
        full_links.add(x)
    return full_links

def msgbox(title, msg, style):
##  Styles:
##  0 : OK
##  1 : OK | Cancel
##  2 : Abort | Retry | Ignore
##  3 : Yes | No | Cancel
##  4 : Yes | No
##  5 : Retry | No 
##  6 : Cancel | Try Again | Continue
     return ctypes.windll.user32.MessageBoxW(0, msg, title, style)
    
    
        
