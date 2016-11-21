from bs4 import BeautifulSoup
from urllib2 import urlopen
import urllib2 as url2
from time import sleep # be nice
import requests
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from selenium.webdriver.support.ui import Select
 
def init_driver():
    driver = webdriver.Firefox()
    driver.wait = WebDriverWait(driver, 5)
    return driver
 
 

 
 
#if __name__ == "__main__":
#    driver = init_driver()
#    lookup(driver, "Selenium")
#    time.sleep(5)
#    driver.quit()


driver = init_driver()
driver.get("http://ndhrhis.com")
#driver.FindElement(By.LinkText("STATISTICAL REPORTS")).Click()
element = driver.find_element_by_link_text('STATISTICAL REPORTS')
time.sleep(1)

element.click()

element2 = driver.find_element_by_link_text('VIEW')
time.sleep(1)
element2.click()
#element = driver.find_element_by_link_text('STATISTICAL REPORTS')
#element.click()

el = driver.find_element_by_name('province')
time.sleep(1)
el.click()

select = Select(driver.find_element_by_name('province'))
#print select.options
opt_vect= [o.text for o in select.options] # these are string-s
print opt_vect



for option in opt_vect:

    select.select_by_visible_text(option)
    option_sel.click()
    #select.select_by_visible_text(opt_vect[5])
    #list_opt.click()
    #stop()
    element2 = driver.find_elements_by_name('submit')

    time.sleep(2)
    element2[3].click()


    select.select_by_visible_text(option)
    element2[3].click()
    #element2 = driver.find_element_by_link_text('VIEW')
    #element2.click()

    #the driver is not updated it pass the wrong page source
    window_before = driver.window_handles[0]
    window_after = driver.window_handles[1]
    driver.switch_to_window(window_after)

    soup = BeautifulSoup(driver.page_source, 'lxml')

    #table = soup.find(class_='RepT')
    #print table

    #stop()











    #stop()

    url = "http://home.strw.leidenuniv.nl/~velliscig/Distribution-By%20Municipality.html"
    #url = 'http://nbviewer.ipython.org/github/chrisalbon/code_py/blob/master/beautiful_soup_scrape_table.ipynb'
    # Create a variable with the URL to this tutorial


    # Scrape the HTML at the url
    r = requests.get(url)

    # Turn the HTML into a Beautiful Soup object
    soup2 = BeautifulSoup(r.text, 'lxml')

    table2 = soup2.find(class_='RepT')
    print table2


    # Create an object of the first object that is class=dataframe
    table = soup.find(class_='RepT')
    print table
    row_vec = table.find_all('tr')
    header = row_vec[0].find_all('td')
    columns_names = [header[i].string.strip() for i in range(1 , len(header))]
    columns_names.insert( 0, u'district')
    columns_names.insert( 1, u'municipality')
    n_col = len(header)
    dict_col = {}
    for col_nam in columns_names:dict_col[col_nam] = []


    #matrix = 
    # Find all the <tr> tag pairs, skip the first one, then for each.
    i_row =0
    for row in table.find_all('tr')[1:]:

        # Create a variable of all the <td> tag pairs in each <tr> tag pair,
        col_in_row = row.find_all('td')
        print '=======index row  ', i_row 
        print '++col_in_row ', col_in_row
        print '\n'    
        n_col_in_row = len(col_in_row)
        print 'n col in row' , n_col_in_row
        # Create a variable of the string inside 1st <td> tag pair,
        if i_row % 2 ==1 :

            column_values_in_row = [float(col_in_row[i].string.strip()) for i in range(1, n_col_in_row)]
            #column_values_in_row = [col_in_row[i].string.strip() for i in range(1,n_col-1)]

            column_values_in_row.insert( 0, prev_tag)
            column_values_in_row.insert( 1, u'cross')
            print '---column_values_in_row  ',column_values_in_row
            print '\n'   



        else:
            #case when 
            column_values_in_row = [col_in_row[i].string.strip() for i in range(n_col_in_row)]
            column_values_in_row[2:]=[float(column_values_in_row[i]) for i in range(2,n_col_in_row)]
            print '^^^^column_values_in_row  ',column_values_in_row
            print '\n'  

            prev_tag = column_values_in_row[0]


        column_values_in_row.insert( 0, option)

        [dict_col[columns_names[i]].append(column_values_in_row[i]) for i in range(n_col)]
        print '***dict', dict_col

        i_row +=1

    
    #if i_row ==3 : stop()
    #dict_col[columns_names[1]].append(column_values_in_row[1])


    # Create a variable of the value of the columns
    driver.close()
    driver.switch_to_window(window_before)

# Create a dataframe from the columns variable
df = pd.DataFrame(dict_col)
