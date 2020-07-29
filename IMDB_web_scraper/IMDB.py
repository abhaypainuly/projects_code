#!/usr/bin/env python
# coding: utf-8

# In[23]:


import requests
from bs4 import BeautifulSoup
import time
from collections import OrderedDict
import logging
import csv
from os import path
import sys


# In[24]:


#Pre-requisits for script!
logging.basicConfig(filename="logfile.log", format='%(asctime)s|%(levelname)s|%(message)s')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
base_url = r'https://www.imdb.com'


# In[25]:


def get_soup(url):
    #Takes the URl and connects with the website
    #Converts the websites data into Beautiful Soup's object
    #Returns the BEautiful Soup's object
    
    logger.info('IN GET SOUP FUNCTION!')
    t = 0     
    while True:
        
        try:
            #Connecting to the websites provided URL
            logger.debug('Requesting the URL!')
            response = requests.get(url)
            response.connection.close()
            logger.debug('{}|{}'.format(response.status_code,url))
            
            #Raising exception in case of status error's
            response.raise_for_status()            
            return BeautifulSoup(response.text,'html.parser')
        
        #Retry in case of connection and timeout error
        except (requests.exceptions.ConnectionError,requests.exceptions.Timeout) as e:            
            #Retrying to connect 10 times after every 6 second
            if t<10:                
                logging.error('Exception: {}'.format(e))           
                time.sleep(6)
                logging.debug('Trying Again.....')
                t += 1
                continue
            #After 10 retry's exiting the system
            logger.error('System Exit,Exception: {}'.format(e))
            raise SystemExit(e)
            
        #In case of other exception
        except requests.exceptions.RequestException as e:
            logger.error('System Exit,Exception: {}'.format(e))
            raise SystemExit(e)           


# In[26]:


def store_state(i,current_url):
    logging.debug('Storing the state of the program!')
    file = open('program_state.txt','w')
    file.write(str(i)+'|'+current_url)
    file.close()
    logging.debug('Succesfully stored the state of the program!')


# In[ ]:





# In[ ]:





# # PHASE1
# Collecting the different movie genres and storing the URLs of enlisting web page!

# And collecting data from those web pages in PHASE2

# In[5]:


logging.info('####################PHASE1###################')

add_url = r'/feature/genre/?ref_=nv_ch_gr'

logging.info('Getting all the Genres and URLs from {}'.format(base_url+add_url))

#Getting the page soup
page_soup = get_soup(base_url+add_url)

#Finding the movies genre 
movie_genres_soup = page_soup.findAll('div',{'class':'ab_links'})

#Finding every cell of the tabble
genres = movie_genres_soup[0].findAll('div',{'class':'table-cell primary'})

#Reading from beautiful Soup object and storing in a CSV file 
#Storing in format {Genre+'|'+URL}
file_name = movie_genres_soup[0].span.span.h3.text.strip()
logging.debug('Writing the Geners and URLs to the "{}" file'.format(file_name))
file = open(file_name+'.csv','w')
for genre in genres:
    file.write(genre.a.text.strip()+'|'+base_url+genre.a.get('href')+'&sort=year,asc'+'\n')
file.close()


# # PHASE2
# Taking the URL's of each genre's from the file generated in PHASE1

# And then scraping all the movies for every list of genres

# In[27]:


#Reading all the Movie Genres URl's from the CSV file and storing in a dictionary
logging.info('####################PHASE2###################')
logging.debug('Reading the Geners and URLs from "Popular Movies by Genre.csv" file! ')

dit = OrderedDict()
genre_url_file = open('Popular Movies by Genre.csv','r')
for line in genre_url_file:
    line = line.split('|')
    dit[line[0]] = line[1]
genre_url_file.close()
logging.debug('Sucessfully created dictionary with Genres and URLs!')

#Creating look up for dictionaries keys:
logging.debug('Creating look up dictionary!')
lookup_keys = list(dit.keys())
logging.debug('Sucessfully created look up dictionary!')


# In[ ]:


logging.info('***************************************************************Starting the Scraper!****************************************************************')
print('Starting the Scraper...........')

#Recovering the previous state of the program
logging.debug('Recovering the previous state of the program')
if path.exists('program_state.txt'):
    file = open('program_state.txt','r')
    file_data = file.readline()
    file.close()
    file_data = file_data.split('|')
    if len(file_data) == 2:
        i = int(file_data[0])
        current_url = file_data[1]
        logging.debug('Successfully recovered the previous state of the program')
    else:
        logging.error('ERROR in "program_state.txt" file.....SystemExit.....')
        print('Scraper exited due to Error in program_state.txt file!')
        sys.exit()
else:
    logging.debug('No previous state found, so starting Scraper from Begining')
    i = 0
    current_url = dit[lookup_keys[i]]

while i <len(lookup_keys):
    
    with open(lookup_keys[i]+'.csv', 'a', newline='\n') as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter='|')
        #csv_file_writer.writerow(['NAME','YEAR','RUNTIME','MIX GENRE','RATING','DISCRIPTION'])        

        while True:

            #Storing the state of program in a file
            store_state(i,current_url)
            
            action_page_soup = get_soup(current_url)

            #Finding the desired TAG in the HTML code
            containers = action_page_soup.findAll('div',{'class':'lister-item-content'})

            #Collecting data from the scrapped web page
            logging.debug('Collecting data from the scraped webpage!')
            page_data = []
            for container in containers:
                
                #Finding the S.No of the movie
                s_no = container.h3.span.text[:-1]
                
                #Finding the Name of the movie
                name = container.h3.a.text
                
                #Finding the Year of the movie if present
                year = container.h3.findAll('span',{'class':'lister-item-year text-muted unbold'})
                if len(year)>0:
                    year = year[0].get_text()[:-1].split('(')[-1]
                else:
                    year = ''
                
                #Finding the Runtime of the movie if present
                runtime = containers[1].p.findAll('span',{'class':'runtime'})
                if len(runtime)>0:
                    runtime = runtime[0].text
                else:
                    runtime = ''
                
                #Finding all the Genres to which the movie belongs
                mix_genre = container.findAll('span',{'class':'genre'})[0].text
                mix_genre = mix_genre[1:].strip()
                
                #Finding the Rating of the movie if present
                rating = container.findAll('div',{'class':'inline-block ratings-imdb-rating'})
                if len(rating)>0:
                    rating = rating[0].strong.text
                else:
                    rating = ''
                
                #Finding the Description of the movie if present and going to the description page
                #if full description is not present
                description = container.findAll('p',{'class':'text-muted'})
                if description[1].text == '\nAdd a Plot\n':
                    description = ''
                elif description[1].a is not None and description[1].a.text.strip() == 'See full summary':
                    plot_soup = get_soup(base_url+description[1].a.get('href'))
                    description = plot_soup.p.text.strip()
                else:
                    description = description[1].text.strip()
        
                page_data.append([s_no,name,year,runtime,mix_genre,rating,description])
        
            #Writing the page data to the CSV file 
            logging.info('Writing the page data to CSV file!')
            csv_file_writer.writerows(page_data)
                       
            #Checking if the page has a next page or not!
            logging.info('Checking for next page if its there!')
            containers = action_page_soup.findAll('div',{'class':'desc'})[0]
            containers = containers.find_all('a')     
            if len(containers)==1:
                if containers[0].text.split('»')[0].strip()!='Next':        
                    break            
                current_url = base_url+containers[0].get('href')
            else:
                current_url = base_url+containers[-1].get('href')
    i += 1            
        
logging.info('Done!')
print('Done!')

