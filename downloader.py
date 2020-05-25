# This script will contain the code that will be executed on a weekly basis
# to download data from the IESO website

from general import *
from link_finder import *
from additional_tools import *
from urllib import request
from time import sleep
import os
 
# Creating my data structure
current_path = os.getcwd()
create_data_files()
create_date_directory(current_path)
s_path = save_path(current_path)

# Setting the page I want to download files from
page = 'http://reports.ieso.ca/public/OntarioZonalDemand/'

# LinkFinder will go to the initial URL and find all of the href links that end
# with '.xml' and store this data. it will also read the file 'Most_Recently_Downloaded.txt'
# and we will compare these sets to find the new data we need to download
try:
    crawler = LinkFinder(page)
except:
    msgbox('Potential Error','Something has gone wrong while trying to access '+ page + '. Applcation Stopped',0)
    exit()

# Extracting the data we need from the LinkFinder object
old_data = crawler.o_data()
new_data = crawler.n_data()

# Finding the values inside the new_data that are not present in the old_data
to_download = set ()
to_download = new_data.difference(old_data)

# Turning the data into urls that can be accessed
download_urls = create_urls(page,to_download)

# Going to the urls and saving the xml data sleep is added as to not bombard the
# site
ticker = 1
full = len(download_urls)
print(str(len(new_data))+' total links found.')
print(str(full)+' new links to be downloaded')
try:
    for url in download_urls:
        sleep(1)
        data = request.urlopen(url).read()
        print('link '+ str(ticker)+'/'+str(full)+' requested data from ' + url)
        ticker += 1
        name = url[len(page):len(url)]
        path = os.path.join(s_path,name)
        store_data(path,data)
except:
    msgbox('Potential Error','Something has gone wrong while downloading xmls files. Application stopped',0)
    exit()

# Saving the downloaded links to the recently downloaded and previously downloaded
# in the recently downloaded it will the links that were available that day
# in the previously downloaded it will keep a complete record of all files that
# were downloaded

clear_file(current_path+'/Most_Recently_Downloaded.txt')
for data in sorted(new_data):
    append_to_file(current_path+'/Most_Recently_Downloaded.txt',data)
for url in sorted(to_download):
    append_to_file(current_path+'/Previously_Downloaded.txt',url)


# creating a message that will notify the user that the download has occured
end_msg = ''
for url in to_download:
    end_msg = end_msg + url +', '

msgbox('Download Completed','Script has successfully downloaded the following links: ' + end_msg,0)


    

        
    






