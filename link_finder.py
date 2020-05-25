# class to identify appropriate links and put them into a set

from html.parser import HTMLParser
from urllib import parse
from urllib import request
from bs4 import BeautifulSoup
from time import sleep
from general import *


class LinkFinder (HTMLParser):

    # the new and old data will be the last part of the links that we need to go to
    # and have already gone, will go do a comparison to see what needs to be downloaded
    def __init__(self,url):
        super().__init__()
        self.url = url
        self.old_data = set()
        self.new_data = set()
        self.old_data = file_to_set('Most_Recently_Downloaded.txt')
        self.html = self.get_html(url)
        self.new_data = self.find_data_links()
        


    # used to get html from a page
    def get_html(self, url):
        html = request.urlopen(url).read()
        return html
    
    # Find all of the current data links, decides if it is a link to an xml
    # and returns the xml links as a set
    def find_data_links(self):
        html = self.html
        links = set()
        soup = BeautifulSoup(html,'html.parser')
        found = soup.find_all('a')
        for a in found:
            ahref = a.get('href')
            chars = len(ahref)
            start = chars-4
            end = chars
            if ahref[start:end] == '.xml':
                links.add(ahref)
        return links

    def n_data(self):
        return self.new_data
    def o_data(self):
        return self.old_data
    def web_html(self):
        return self.html

    

    
    
