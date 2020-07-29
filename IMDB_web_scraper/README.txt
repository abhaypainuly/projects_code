Objective:
  Scrapes all the movies data and stores the data in a csv file genre wise.
  
 Key Features:
  Logging system to debug errors
  Resuming capacity of the program to start scraping data from where it have left last time
  Automatically scraping mutiple pages if present and storing and stoping accordingly
  

Information about the files:
  Data                   ---> contains the scraped data  
  logfile.log            ---> contains all the loging data
  IMDB.ipynb             ---> contains the source code
  programming_state.txt  ---> contains the state from which the program has to resume if it is restarted again.

Programming Language
  Python

Modules Used:
  Requests Module
  Beautiful Soup
  Logging
  CSV
  
 Working:
  Project is divided in to two phases.
  
  PHASE1
  Collecting the list of different movie genres and storing the Name and URLs of web page leading to the data about the movies belonding to those genre.
  
  PHASE2
  Taking the URL's of each genre's from the csv file generated in PHASE1
  And then scraping all the movies data for every list of genres and storing in csv file according to the genere 
  
