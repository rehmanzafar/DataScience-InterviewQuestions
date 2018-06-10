## Edureka
### Q1. What are the key features of Python?
Ans: These are the few key features of Python:
- Python is an interpreted language. That means that, unlike languages like C and its variants, Python does not need to be compiled before it is run. Other interpreted languages include PHP and Ruby.
- Python is dynamically typed, this means that you don’t need to state the types of variables when you declare them or anything like that. You can do things like x=111 and then x="I'm a string" without error
- Python is well suited to object orientated programming in that it allows the definition of classes along with composition and inheritance. Python does not have access specifiers (like C++’s public, private), the justification for this point is given as “we are all adults here”
- In Python, functions are first-class objects. This means that they can be assigned to variables, returned from other functions and passed into functions. Classes are also first class objects
- Writing Python code is quick but running it is often slower than compiled languages. Fortunately，Python allows the inclusion of C based extensions so bottlenecks can be optimized away and often are. The numpy package is a good example of this, it’s really quite quick because a lot of the number crunching it does isn’t actually done by Python
- Python finds use in many spheres – web applications, automation, scientific modelling, big data applications and many more. It’s also often used as “glue” code to get other languages and components to play nice.
### Q2. What is the difference between deep and shallow copy?
Ans: Shallow copy is used when a new instance type gets created and it keeps the values that are copied in the new instance. Shallow copy is used to copy the reference pointers just like it copies the values. These references point to the original objects and the changes made in any member of the class will also affect the original copy of it. Shallow copy allows faster execution of the program and it depends on the size of the data that is used.
Deep copy is used to store the values that are already copied. Deep copy doesn’t copy the reference pointers to the objects. It makes the reference to an object and the new object that is pointed by some other object gets stored. The changes made in the original copy won’t affect any other copy that uses the object. Deep copy makes execution of the program slower due to making certain copies for each object that is been called.
### Q3. What is the difference between list and tuples?
Ans: Lists are mutable i.e they can be edited. Syntax: list_1 = [10, ‘Chelsea’, 20]
Tuples are immutable (tuples are lists which can’t be edited). Syntax: tup_1 = (10, ‘Chelsea’ , 20)
### Q4. How is Multithreading achieved in Python?
Ans: 
1. Python has a multi-threading package but if you want to multi-thread to speed your code up.
2. Python has a construct called the Global Interpreter Lock (GIL). The GIL makes sure that only one of your ‘threads’ can execute at any one time. A thread acquires the GIL, does a little work, then passes the GIL onto the next thread.
3. This happens very quickly so to the human eye it may seem like your threads are executing in parallel, but they are really just taking turns using the same CPU core.
4. All this GIL passing adds overhead to execution. This means that if you want to make your code run faster then using the threading package often isn’t a good idea.
### Q5. How can the ternary operators be used in python?
Ans: The Ternary operator is the operator that is used to show the conditional statements. This consists of the true or false values with a statement that has to be evaluated for it.
Syntax:
The Ternary operator will be given as:
[on_true] if [expression] else [on_false]x, y = 25, 50big = x if x < y else y
Example:
The expression gets evaluated like if x<y else y, in this case if x<y is true then the value is returned as big=x and if it is incorrect then big=y will be sent as a result.
### Q6. How is memory managed in Python?
Ans: 
1. Python memory is managed by Python private heap space. All Python objects and data structures are located in a private heap. The programmer does not have an access to this private heap and interpreter takes care of this Python private heap. 
2. The allocation of Python heap space for Python objects is done by Python memory manager. The core API gives access to some tools for the programmer to code.
3. Python also have an inbuilt garbage collector, which recycle all the unused memory and frees the memory and makes it available to the heap space.
### Q7. Explain Inheritance in Python with an example.
Ans: Inheritance allows One class to gain all the members(say attributes and methods) of another class. Inheritance provides code reusability, makes it easier to create and maintain an application. The class from which we are inheriting is called super-class and the class that is inherited is called a derived / child class.
They are different types of inheritance supported by Python:
1.	Single Inheritance – where a derived class acquires the members of a single super class.
2.	Multi-level inheritance – a derived class d1 in inherited from base class base1, and d2 is inherited from base2.
3.	Hierarchical inheritance – from one base class you can inherit any number of child classes
4.	Multiple inheritance – a derived class is inherited from more than one base class.
### Q8. Explain what Flask is and its benefits?
Ans: Flask is a web micro framework for Python based on “Werkzeug, Jinja2 and good intentions” BSD license. Werkzeug and Jinja2 are two of its dependencies. This means it will have little to no dependencies on external libraries.  It makes the framework light while there is little dependency to update and less security bugs.
A session basically allows you to remember information from one request to another. In a flask, a session uses a signed cookie so the user can look at the session contents and modify. The user can modify the session if only it has the secret key Flask.secret_key.
### Q9. What is the usage of help() and dir() function in Python?
Ans: Help() and dir() both functions are accessible from the Python interpreter and used for viewing a consolidated dump of built-in functions. 
1.	Help() function: The help() function is used to display the documentation string and also facilitates you to see the help related to modules, keywords, attributes, etc.
2.	Dir() function: The dir() function is used to display the defined symbols.
### Q10. Whenever Python exits, why isn’t all the memory de-allocated?
Ans: 
1.	Whenever Python exits, especially those Python modules which are having circular references to other objects or the objects that are referenced from the global namespaces are not always de-allocated or freed.
2.	It is impossible to de-allocate those portions of memory that are reserved by the C library.
3.	On exit, because of having its own efficient clean up mechanism, Python would try to de-allocate/destroy every other object.
### Q11. What is dictionary in Python?
Ans: The built-in datatypes in Python is called dictionary. It defines one-to-one relationship between keys and values. Dictionaries contain pair of keys and their corresponding values. Dictionaries are indexed by keys.
Let’s take an example:
The following example contains some keys. Country, Capital & PM. Their corresponding values are India, Delhi and Modi respectively.
```python
dict={'Country':'Canada','Capital':'Ottawa'}
print dict[Country]
Canada
print dict[Capital]
Ottawa
```
### Q12. What is monkey patching in Python?
Ans: In Python, the term monkey patch only refers to dynamic modifications of a class or module at run-time.
Consider the below example:
```python
# m.py
class MyClass:
def f(self):
print "f()"

We can then run the monkey-patch testing like this:
import m
def monkey_f(self):
print "monkey_f()"
 
m.MyClass.f = monkey_f
obj = m.MyClass()
obj.f()

```
The output will be as below:
monkey_f()

As we can see, we did make some changes in the behavior of f() in MyClass using the function we defined, monkey_f(), outside of the module m.

### Q13. What does this mean: *args, **kwargs? And why would we use it?
Ans: We use *args when we aren’t sure how many arguments are going to be passed to a function, or if we want to pass a stored list or tuple of arguments to a function. **kwargsis used when we don’t know how many keyword arguments will be passed to a function, or it can be used to pass the values of a dictionary as keyword arguments. The identifiers args and kwargs are a convention, you could also use *bob and **billy but that would not be wise.

### Q14. Write a one-liner that will count the number of capital letters in a file. Your code should work even if the file is too big to fit in memory.
Ans:  Let us first write a multiple line solution and then convert it to one liner code.
```python
with open(SOME_LARGE_FILE) as fh:
count = 0
text = fh.read()
for character in text:
    if character.isupper():
count += 1
```
We will now try to transform this into a single line.
```python
count sum(1 for line in fh for character in line if character.isupper())
```
### Q15. What are negative indexes and why are they used?
Ans: The sequences in Python are indexed and it consists of the positive as well as negative numbers. The numbers that are positive uses ‘0’ that is uses as first index and ‘1’ as the second index and the process goes on like that.
The index for the negative number starts from ‘-1’ that represents the last index in the sequence and ‘-2’ as the penultimate index and the sequence carries forward like the positive number.
The negative index is used to remove any new-line spaces from the string and allow the string to except the last character that is given as S[:-1]. The negative index is also used to show the index to represent the string in correct order.
### Q16. How can you randomize the items of a list in place in Python?
Ans: Consider the example shown below:
```python
from random import shuffle
x = ['Keep', 'The', 'Blue', 'Flag', 'Flying', 'High']
shuffle(x)
print(x)
```
The output of the following code is as below.
['Flying', 'Keep', 'Blue', 'High', 'The', 'Flag']

### Q17. What is the process of compilation and linking in python?
Ans: The compiling and linking allows the new extensions to be compiled properly without any error and the linking can be done only when it passes the compiled procedure. If the dynamic loading is used then it depends on the style that is being provided with the system. The python interpreter can be used to provide the dynamic loading of the configuration setup files and will rebuild the interpreter.
The steps that is required in this as:
1.	Create a file with any name and in any language that is supported by the compiler of your system. For example file.c or file.cpp
2.	Place this file in the Modules/ directory of the distribution which is getting used.
3.	Add a line in the file Setup.local that is present in the Modules/ directory.
4.	Run the file using spam file.o
5.	After successful run of this rebuild the interpreter by using the make command on the top-level directory.
6.	If the file is changed then run rebuildMakefile by using the command as ‘make Makefile’.
### Q18. Write a sorting algorithm for a numerical dataset in Python.
Ans: The following code can be used to sort a list in Python: 
```python
list = ["1", "4", "0", "6", "9"] 
list = [int(i) for i in list] 
list.sort()
print (list)
```

### Q19. Looking at the below code, write down the final values of A0, A1, …An.
Ans: The following will be the final outputs of A0, A1, … A6
```python
A0 = dict(zip(('a','b','c','d','e'),(1,2,3,4,5)))
A1 = range(10)A2 = sorted([i for i in A1 if i in A0])
A3 = sorted([A0[s] for s in A0])
A4 = [i for i in A1 if i in A3]
A5 = {i:i*i for i in A1}
A6 = [[i,i*i] for i in A1]
print(A0,A1,A2,A3,A4,A5,A6)
```
A0 = {'a': 1, 'c': 3, 'b': 2, 'e': 5, 'd': 4} # the order may vary
A1 = range(0, 10) 
A2 = []
A3 = [1, 2, 3, 4, 5]
A4 = [1, 2, 3, 4, 5]
A5 = {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}
A6 = [[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81]]
### Q20. Explain split(), sub(), subn() methods of “re” module in Python.
Ans: To modify the strings, Python’s “re” module is providing 3 methods. They are:
•	split() – uses a regex pattern to “split” a given string into a list.
•	sub() – finds all substrings where the regex pattern matches and then replace them with a different string
•	subn() – it is similar to sub() and also returns the new string along with the no. of replacements.
### Q21. How can you generate random numbers in Python?
Ans: Random module is the standard module that is used to generate the random number. The method is defined as:
```python
import random
random.random
```
The statement random.random() method return the floating point number that is in the range of [0, 1). The function generates the random float numbers. The methods that are used with the random class are the bound methods of the hidden instances. The instances of the Random can be done to show the multi-threading programs that creates different instance of individual threads. The other random generators that are used in this are:
1.	randrange(a, b): it chooses an integer and define the range in-between [a, b). It returns the elements by selecting it randomly from the range that is specified. It doesn’t build a range object.
2.	uniform(a, b): it chooses a floating point number that is defined in the range of [a,b).Iyt returns the floating point number
3.	normalvariate(mean, sdev): it is used for the normal distribution where the mu is a mean and the sdev is a sigma that is used for standard deviation.
4.	The Random class that is used and instantiated creates an independent multiple random number generators.
### Q22. What is the difference between range & xrange?
Ans: For the most part, xrange and range are the exact same in terms of functionality. They both provide a way to generate a list of integers for you to use, however you please. The only difference is that range returns a Python list object and x range returns an xrange object.
This means that xrange doesn’t actually generate a static list at run-time like range does. It creates the values as you need them with a special technique called yielding. This technique is used with a type of object known as generators. That means that if you have a really gigantic range you’d like to generate a list for, say one billion, xrange is the function to use.
This is especially true if you have a really memory sensitive system such as a cell phone that you are working with, as range will use as much memory as it can to create your array of integers, which can result in a Memory Error and crash your program. It’s a memory hungry beast.
### Q23. What is pickling and unpickling?
Ans: Pickle module accepts any Python object and converts it into a string representation and dumps it into a file by using dump function, this process is called pickling. While the process of retrieving original Python objects from the stored string representation is called unpickling.
## Django – Python Interview Questions
### Q24. Mention the differences between Django, Pyramid and Flask.
Ans: 
- Flask is a “microframework” primarily build for a small application with simpler requirements. In flask, you have to use external libraries. Flask is ready to use.
- Pyramid is built for larger applications. It provides flexibility and lets the developer use the right tools for their project. The developer can choose the database, URL structure, templating style and more. Pyramid is heavy configurable.
- Django can also used for larger applications just like Pyramid. It includes an ORM.
### Q25. Discuss the Django architecture.
Ans: Django MVT Pattern:

![picture alt]( https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2017/06/Django-Architecture-Python-Interview-Questions-Edureka.png)

The developer provides the Model, the view and the template then just maps it to a URL and Django does the magic to serve it to the user.

### Q26. Explain how you can set up the Database in Django.
Ans: You can use the command edit mysite/setting.py , it is a normal python module with module level representing Django settings.

Django uses SQLite by default; it is easy for Django users as such it won’t require any other type of installation. In the case your database choice is different that you have to the following keys in the DATABASE ‘default’ item to match your database connection settings.

Engines: you can change database by using ‘django.db.backends.sqlite3’ , ‘django.db.backeneds.mysql’, ‘django.db.backends.postgresql_psycopg2’, ‘django.db.backends.oracle’ and so on
Name: The name of your database. In the case if you are using SQLite as your database, in that case database will be a file on your computer, Name should be a full absolute path, including file name of that file.
If you are not choosing SQLite as your database then settings like Password, Host, User, etc. must be added.
Django uses SQLite as default database, it stores data as a single file in the filesystem. If you do have a database server—PostgreSQL, MySQL, Oracle, MSSQL—and want to use it rather than SQLite, then use your database’s administration tools to create a new database for your Django project. Either way, with your (empty) database in place, all that remains is to tell Django how to use it. This is where your project’s settings.py file comes in.

We will add the following lines of code to the setting.py file:
```python
DATABASES = {
     'default': {
          'ENGINE' : 'django.db.backends.sqlite3',
          'NAME' : os.path.join(BASE_DIR, 'db.sqlite3'),
     }
}
```

### Q27. Give an example how you can write a VIEW in Django?
Ans: This is how we can use write a view in Django:
```python
from django.http import HttpResponse
import datetime
 
def Current_datetime(request):
     now = datetime.datetime.now()
     html = "<html><body>It is now %s</body></html>" % now
     return HttpResponse(html)
```
Returns the current date and time, as an HTML document

### Q28. Mention what the Django templates consists of.
Ans: The template is a simple text file.  It can create any text-based format like XML, CSV, HTML, etc.  A template contains variables that get replaced with values when the template is evaluated and tags (% tag %) that controls the logic of the template.
![picture alt](https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2017/06/Django-Template-Python-Interview-Questions-Edureka.png)

### Q29. Explain the use of session in Django framework?
Ans: Django provides session that lets you store and retrieve data on a per-site-visitor basis. Django abstracts the process of sending and receiving cookies, by placing a session ID cookie on the client side, and storing all the related data on the server side.

![picture alt]( https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2017/06/Django-Framework-Python-Interview-Questions-Edureka.png)

So the data itself is not stored client side. This is nice from a security perspective.

### Q30. List out the inheritance styles in Django.
Ans: In Django, there is three possible inheritance styles:

Abstract Base Classes: This style is used when you only wants parent’s class to hold information that you don’t want to type out for each child model.
Multi-table Inheritance: This style is used If you are sub-classing an existing model and need each model to have its own database table.
Proxy models: You can use this model, If you only want to modify the Python level behavior of the model, without changing the model’s fields.
Web Scraping – Python Interview Questions
### Q31. How To Save An Image Locally Using Python Whose URL Address I Already Know?
Ans: We will use the following code to save an image locally from an URL address
```python
import urllib.request
urllib.request.urlretrieve("URL", "local-filename.jpg")
```
### Q32. How can you Get the Google cache age of any URL or web page?
Ans: Use the following URL format:

http://webcache.googleusercontent.com/search?q=cache:URLGOESHERE

Be sure to replace “URLGOESHERE” with the proper web address of the page or site whose cache you want to retrieve and see the time for. For example, to check the Google Webcache age of edureka.co you’d use the following URL:

http://webcache.googleusercontent.com/search?q=cache:edureka.co

### Q33. You are required to scrap data from IMDb top 250 movies page. It should only have fields movie name, year, and rating.
Ans: We will use the following lines of code:
```python
from bs4 import BeautifulSoup
 import requests
import sys
url = 'http://www.imdb.com/chart/top'
response = requests.get(url)
soup = BeautifulSoup(response.text)
tr = soup.findChildren("tr")
tr = iter(tr)
next(tr)
 
for movie in tr:
title = movie.find('td', {'class': 'titleColumn'} ).find('a').contents[0]
year = movie.find('td', {'class': 'titleColumn'} ).find('span', {'class': 'secondaryInfo'}).contents[0]
rating = movie.find('td', {'class': 'ratingColumn imdbRating'} ).find('strong').contents[0]
row = title + ' - ' + year + ' ' + ' ' + rating
 
print(row)
```
The above code will help scrap data from IMDb’s top 250 list

Data Analysis – Python Interview Questions
### Q34. What is map function in Python?
Ans: map function executes the function given as the first argument on all the elements of the iterable given as the second argument. If the function given takes in more than 1 arguments, then many iterables are given. #Follow the link to know more similar functions.

### Q35. How to get indices of N maximum values in a NumPy array?
Ans: We can get the indices of N maximum values in a NumPy array using the below code:
```python
import numpy as np
arr = np.array([1, 3, 2, 4, 5])
print(arr.argsort()[-3:][::-1])
```
Output
[4, 5, 1]
### Q36. How do you calculate percentiles with Python/ NumPy?
Ans: We can calculate percentiles with the following code
```python
import numpy as np
a = np.array([1,2,3,4,5])
p = np.percentile(a, 50)  #Returns 50th percentile, e.g. median
print(p)
```
Output
3
### Q37. What advantages do NumPy arrays offer over (nested) Python lists?
Ans: 

Python’s lists are efficient general-purpose containers. They support (fairly) efficient insertion, deletion, appending, and concatenation, and Python’s list comprehensions make them easy to construct and manipulate.
They have certain limitations: they don’t support “vectorized” operations like elementwise addition and multiplication, and the fact that they can contain objects of differing types mean that Python must store type information for every element, and must execute type dispatching code when operating on each element.
NumPy is not just more efficient; it is also more convenient. You get a lot of vector and matrix operations for free, which sometimes allow one to avoid unnecessary work. And they are also efficiently implemented.
NumPy array is faster and You get a lot built in with NumPy, FFTs, convolutions, fast searching, basic statistics, linear algebra, histograms, etc. 
### Q38. Explain the use of decorators.
Ans: Decorators in Python are used to modify or inject code in functions or classes. Using decorators, you can wrap a class or function method call so that a piece of code can be executed before or after the execution of the original code. Decorators can be used to check for permissions, modify or track the arguments passed to a method, logging the calls to a specific method, etc.

### Q39. What is the difference between NumPy and SciPy?
Ans: In an ideal world, NumPy would contain nothing but the array data type and the most basic operations: indexing, sorting, reshaping, basic elementwise functions, et cetera.
All numerical code would reside in SciPy. However, one of NumPy’s important goals is compatibility, so NumPy tries to retain all features supported by either of its predecessors.
Thus NumPy contains some linear algebra functions, even though these more properly belong in SciPy. In any case, SciPy contains more fully-featured versions of the linear algebra modules, as well as many other numerical algorithms.
If you are doing scientific computing with python, you should probably install both NumPy and SciPy. Most new features belong in SciPy rather than NumPy.
Q40. How do you make 3D plots/visualizations using NumPy/SciPy?
Ans: Like 2D plotting, 3D graphics is beyond the scope of NumPy and SciPy, but just as in the 2D case, packages exist that integrate with NumPy. Matplotlib provides basic 3D plotting in the mplot3d subpackage, whereas Mayavi provides a wide range of high-quality 3D visualization features, utilizing the powerful VTK engine.

## Multiple Choice Questions
### Q41. Which of the following statements create a dictionary? (Multiple Correct Answers Possible)
a) d = {}
b) d = {“john”:40, “peter”:45}
c) d = {40:”john”, 45:”peter”}
d) d = (40:”john”, 45:”50”)
Answer: b, c & d. 

Dictionaries are created by specifying keys and values.

### Q42. Which one of these is floor division?
a) /
b) //
c) %
d) None of the mentioned
Answer: b) //

When both of the operands are integer then python chops out the fraction part and gives you the round off value, to get the accurate answer use floor division. For ex, 5/2 = 2.5 but both of the operands are integer so answer of this expression in python is 2. To get the 2.5 as the answer, use floor division using //. So, 5//2 = 2.5

### Q43. What is the maximum possible length of an identifier?
a) 31 characters
b) 63 characters
c) 79 characters
d) None of the above
Answer: d) None of the above

Identifiers can be of any length.

### Q44. Why are local variable names beginning with an underscore discouraged?
a) they are used to indicate a private variables of a class
b) they confuse the interpreter
c) they are used to indicate global variables
d) they slow down execution
Answer: a) they are used to indicate a private variables of a class

As Python has no concept of private variables, leading underscores are used to indicate variables that must not be accessed from outside the class.

### Q45. Which of the following is an invalid statement?
a) abc = 1,000,000
b) a b c = 1000 2000 3000
c) a,b,c = 1000, 2000, 3000
d) a_b_c = 1,000,000
Answer: b) a b c = 1000 2000 3000

Spaces are not allowed in variable names.

### Q46. What is the output of the following?
try:
    if '1' != 1:
        raise "someError"
    else:
        print("someError has not occured")
except "someError":
    print ("someError has occured")
a) someError has occured
b) someError has not occured
c) invalid code
d) none of the above
Answer: c) invalid code

A new exception class must inherit from a BaseException. There is no such inheritance here.

### Q47. Suppose list1 is [2, 33, 222, 14, 25], What is list1[-1] ?
a) Error
b) None
c) 25
d) 2
Answer: c) 25

The index -1 corresponds to the last index in the list.

### Q48. To open a file c:\scores.txt for writing, we use
a) outfile = open(“c:\scores.txt”, “r”)
b) outfile = open(“c:\\scores.txt”, “w”)
c) outfile = open(file = “c:\scores.txt”, “r”)
d) outfile = open(file = “c:\\scores.txt”, “o”)
Answer: b) The location contains double slashes ( \\ ) and w is used to indicate that file is being written to.

### Q49. What is the output of the following?
```python
f = None
 for i in range (5):
    with open("data.txt", "w") as f:
        if i > 2:
            break
 
print f.closed
```
a) True
b) False
c) None
d) Error
Answer: a) True 

The WITH statement when used with open file guarantees that the file object is closed when the with block exits.

### Q50. When will the else part of try-except-else be executed?
a) always
b) when an exception occurs
c) when no exception occurs
d) when an exception occurs in to except block
Answer: c) when no exception occurs

The else part is executed when no exception occurs.

## codementor
### Question 1 What is Python really? You can (and are encouraged) make comparisons to other technologies in your answer
Answer
Here are a few key points:
- Python is an interpreted language. That means that, unlike languages like C and its variants, Python does not need to be compiled before it is run. Other interpreted languages include PHP and Ruby.
- Python is dynamically typed, this means that you don't need to state the types of variables when you declare them or anything like that. You can do things like x=111 and then x="I'm a string" without error
- Python is well suited to object orientated programming in that it allows the definition of classes along with composition and inheritance. Python does not have access specifiers (like C++'s public, private), the justification for this point is given as "we are all adults here"
- In Python, functions are first-class objects. This means that they can be assigned to variables, returned from other functions and passed into functions. Classes are also first class objects
- Writing Python code is quick but running it is often slower than compiled languages. Fortunately， Python allows the inclusion of C based extensions so bottlenecks can be optimised away and often are. The numpy package is a good example of this, it's really quite quick because a lot of the number crunching it does isn't actually done by Python
- Python finds use in many spheres - web applications, automation, scientific modelling, big data applications and many more. It's also often used as "glue" code to get other languages and components to play nice.
- Python makes difficult things easy so programmers can focus on overriding algorithms and structures rather than nitty-gritty low level details.
Why This Matters:
If you are applying for a Python position, you should know what it is and why it is so gosh-darn cool. And why it isn't o.O
### Question 2 Fill in the missing code:
```python
def print_directory_contents(sPath):
    """
    This function takes the name of a directory and prints out the paths files within that 
    directory as well as any files contained in 
    contained directories. 

    This function is similar to os.walk. Please don't
    use os.walk in your answer. We are interested in your 
    ability to work with nested structures. 
    """
    fill_this_in
```
Answer
```python
def print_directory_contents(sPath):
    import os                                       
    for sChild in os.listdir(sPath):                
        sChildPath = os.path.join(sPath,sChild)
        if os.path.isdir(sChildPath):
            print_directory_contents(sChildPath)
        else:
            print(sChildPath)
```

Pay Special Attention
- Be consistent with your naming conventions. If there is a naming convention evident in any sample code, stick to it. Even if it is not the naming convention you usually use
- Recursive functions need to recurse and terminate. Make sure you understand how this happens so that you avoid bottomless callstacks
- We use the os module for interacting with the operating system in a way that is cross platform. You could say sChildPath = sPath + '/' + sChild but that wouldn't work on windows
- Familiarity with base packages is really worthwhile, but don't break your head trying to memorize everything, Google is your friend in the workplace!
- Ask questions if you don't understand what the code is supposed to do
- KISS! Keep it Simple, Stupid!
Why This Matters:
- Displays knowledge of basic operating system interaction stuff
- Recursion is hella useful
### Question 3 Looking at the below code, write down the final values of A0, A1, ...An.
```python
A0 = dict(zip(('a','b','c','d','e'),(1,2,3,4,5)))
A1 = range(10)
A2 = sorted([i for i in A1 if i in A0])
A3 = sorted([A0[s] for s in A0])
A4 = [i for i in A1 if i in A3]
A5 = {i:i*i for i in A1}
A6 = [[i,i*i] for i in A1]
```
If you dont know what zip is don't stress out. No sane employer will expect you to memorize the standard library. Here is the output of help(zip).
```python
zip(...)
    zip(seq1 [, seq2 [...]]) -> [(seq1[0], seq2[0] ...), (...)]
    
    Return a list of tuples, where each tuple contains the i-th element
    from each of the argument sequences.  The returned list is truncated
    in length to the length of the shortest argument sequence.
```
If that doesn't make sense then take a few minutes to figure it out however you choose to.
Answer
A0 = {'a': 1, 'c': 3, 'b': 2, 'e': 5, 'd': 4}  # the order may vary
A1 = range(0, 10) # or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in python 2
A2 = []
A3 = [1, 2, 3, 4, 5]
A4 = [1, 2, 3, 4, 5]
A5 = {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}
A6 = [[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81]]
Why This Matters
1.	List comprehension is a wonderful time saver and a big stumbling block for a lot of people
2.	If you can read them, you can probably write them down
3.	Some of this code was made to be deliberately weird. You may need to work with some weird people
### Question 4 Python and multi-threading. Is it a good idea? List some ways to get some Python code to run in a parallel way.
Answer
Python doesn't allow multi-threading in the truest sense of the word. It has a multi-threading package but if you want to multi-thread to speed your code up, then it's usually not a good idea to use it. Python has a construct called the Global Interpreter Lock (GIL). The GIL makes sure that only one of your 'threads' can execute at any one time. A thread acquires the GIL, does a little work, then passes the GIL onto the next thread. This happens very quickly so to the human eye it may seem like your threads are executing in parallel, but they are really just taking turns using the same CPU core. All this GIL passing adds overhead to execution. This means that if you want to make your code run faster then using the threading package often isn't a good idea.
There are reasons to use Python's threading package. If you want to run some things simultaneously, and efficiency is not a concern, then it's totally fine and convenient. Or if you are running code that needs to wait for something (like some IO) then it could make a lot of sense. But the threading library won't let you use extra CPU cores.
Multi-threading can be outsourced to the operating system (by doing multi-processing), some external application that calls your Python code (eg, Spark or Hadoop), or some code that your Python code calls (eg: you could have your Python code call a C function that does the expensive multi-threaded stuff).
Why This Matters
Because the GIL is an A-hole. Lots of people spend a lot of time trying to find bottlenecks in their fancy Python multi-threaded code before they learn what the GIL is.
### Question 5
How do you keep track of different versions of your code?
Answer:
Version control! At this point, you should act excited and tell them how you even use Git (or whatever is your favorite) to keep track of correspondence with Granny. Git is my preferred version control system, but there are others, for example subversion.
Why This Matters:
Because code without version control is like coffee without a cup. Sometimes we need to write once-off throw away scripts and that's ok, but if you are dealing with any significant amount of code, a version control system will be a benefit. Version Control helps with keeping track of who made what change to the code base; finding out when bugs were introduced to the code; keeping track of versions and releases of your software; distributing the source code amongst team members; deployment and certain automations. It allows you to roll your code back to before you broke it which is great on its own. Lots of stuff. It's just great.
### Question 6 What does this code output:
```python
def f(x,l=[]):
    for i in range(x):
        l.append(i*i)
    print(l) 

f(2)
f(3,[3,2,1])
f(3)
```
Answer
[0, 1]
[3, 2, 1, 0, 1, 4]
[0, 1, 0, 1, 4]
Hu?
The first function call should be fairly obvious, the loop appends 0 and then 1 to the empty list, l. l is a name for a variable that points to a list stored in memory.
The second call starts off by creating a new list in a new block of memory. l then refers to this new list. It then appends 0, 1 and 4 to this new list. So that's great.
The third function call is the weird one. It uses the original list stored in the original memory block. That is why it starts off with 0 and 1.
Try this out if you don't understand:
```python
l_mem = []

l = l_mem           # the first call
for i in range(2):
    l.append(i*i)

print(l)            # [0, 1]

l = [3,2,1]         # the second call
for i in range(3):
    l.append(i*i)

print(l)            # [3, 2, 1, 0, 1, 4]

l = l_mem           # the third call
for i in range(3):
    l.append(i*i)

print(l)            # [0, 1, 0, 1, 4]
```
### Question 7 What is monkey patching and is it ever a good idea?
Answer
Monkey patching is changing the behaviour of a function or object after it has already been defined. For example:
```python 
import datetime
datetime.datetime.now = lambda: datetime.datetime(2012, 12, 12)
```
Most of the time it's a pretty terrible idea - it is usually best if things act in a well-defined way. One reason to monkey patch would be in testing. The mock package is very useful to this end.
Why This Matters
It shows that you understand a bit about methodologies in unit testing. Your mention of monkey avoidance will show that you aren't one of those coders who favor fancy code over maintainable code (they are out there, and they suck to work with). Remember the principle of KISS? And it shows that you know a little bit about how Python works on a lower level, how functions are actually stored and called and suchlike.
PS: it's really worth reading a little bit about mock if you haven't yet. It's pretty useful.
### Question 8 What does this stuff mean: *args, **kwargs? And why would we use it?
Answer
Use *args when we aren't sure how many arguments are going to be passed to a function, or if we want to pass a stored list or tuple of arguments to a function. **kwargs is used when we dont know how many keyword arguments will be passed to a function, or it can be used to pass the values of a dictionary as keyword arguments. The identifiers args and kwargs are a convention, you could also use *bob and **billy but that would not be wise.
Here is a little illustration:
```python
def f(*args,**kwargs): print(args, kwargs)

l = [1,2,3]
t = (4,5,6)
d = {'a':7,'b':8,'c':9}

f()
f(1,2,3)                    # (1, 2, 3) {}
f(1,2,3,"groovy")           # (1, 2, 3, 'groovy') {}
f(a=1,b=2,c=3)              # () {'a': 1, 'c': 3, 'b': 2}
f(a=1,b=2,c=3,zzz="hi")     # () {'a': 1, 'c': 3, 'b': 2, 'zzz': 'hi'}
f(1,2,3,a=1,b=2,c=3)        # (1, 2, 3) {'a': 1, 'c': 3, 'b': 2}

f(*l,**d)                   # (1, 2, 3) {'a': 7, 'c': 9, 'b': 8}
f(*t,**d)                   # (4, 5, 6) {'a': 7, 'c': 9, 'b': 8}
f(1,2,*t)                   # (1, 2, 4, 5, 6) {}
f(q="winning",**d)          # () {'a': 7, 'q': 'winning', 'c': 9, 'b': 8}
f(1,2,*t,q="winning",**d)   # (1, 2, 4, 5, 6) {'a': 7, 'q': 'winning', 'c': 9, 'b': 8}

def f2(arg1,arg2,*args,**kwargs): print(arg1,arg2, args, kwargs)

f2(1,2,3)                       # 1 2 (3,) {}
f2(1,2,3,"groovy")              # 1 2 (3, 'groovy') {}
f2(arg1=1,arg2=2,c=3)           # 1 2 () {'c': 3}
f2(arg1=1,arg2=2,c=3,zzz="hi")  # 1 2 () {'c': 3, 'zzz': 'hi'}
f2(1,2,3,a=1,b=2,c=3)           # 1 2 (3,) {'a': 1, 'c': 3, 'b': 2}

f2(*l,**d)                   # 1 2 (3,) {'a': 7, 'c': 9, 'b': 8}
f2(*t,**d)                   # 4 5 (6,) {'a': 7, 'c': 9, 'b': 8}
f2(1,2,*t)                   # 1 2 (4, 5, 6) {}
f2(1,1,q="winning",**d)      # 1 1 () {'a': 7, 'q': 'winning', 'c': 9, 'b': 8}
f2(1,2,*t,q="winning",**d)   # 1 2 (4, 5, 6) {'a': 7, 'q': 'winning', 'c': 9, 'b': 8} 
```
Why Care?
Sometimes we will need to pass an unknown number of arguments or keyword arguments into a function. Sometimes we will want to store arguments or keyword arguments for later use. Sometimes it's just a time saver.
### Question 9 What do these mean to you: @classmethod, @staticmethod, @property?
Answer Background Knowledge
These are decorators. A decorator is a special kind of function that either takes a function and returns a function, or takes a class and returns a class. The @ symbol is just syntactic sugar that allows you to decorate something in a way that's easy to read.
```python
@my_decorator
def my_func(stuff):
    do_things
```
Is equivalent to
```python
def my_func(stuff):
    do_things

my_func = my_decorator(my_func)
```
You can find a tutorial on how decorators in general work here.
Actual Answer
The decorators @classmethod, @staticmethod and @property are used on functions defined within classes. Here is how they behave:
```python 
class MyClass(object):
    def __init__(self):
        self._some_property = "properties are nice"
        self._some_other_property = "VERY nice"
    def normal_method(*args,**kwargs):
        print("calling normal_method({0},{1})".format(args,kwargs))
    @classmethod
    def class_method(*args,**kwargs):
        print("calling class_method({0},{1})".format(args,kwargs))
    @staticmethod
    def static_method(*args,**kwargs):
        print("calling static_method({0},{1})".format(args,kwargs))
    @property
    def some_property(self,*args,**kwargs):
        print("calling some_property getter({0},{1},{2})".format(self,args,kwargs))
        return self._some_property
    @some_property.setter
    def some_property(self,*args,**kwargs):
        print("calling some_property setter({0},{1},{2})".format(self,args,kwargs))
        self._some_property = args[0]
    @property
    def some_other_property(self,*args,**kwargs):
        print("calling some_other_property getter({0},{1},{2})".format(self,args,kwargs))
        return self._some_other_property

o = MyClass()
# undecorated methods work like normal, they get the current instance (self) as the first argument

o.normal_method 
# <bound method MyClass.normal_method of <__main__.MyClass instance at 0x7fdd2537ea28>>

o.normal_method() 
# normal_method((<__main__.MyClass instance at 0x7fdd2537ea28>,),{})

o.normal_method(1,2,x=3,y=4) 
# normal_method((<__main__.MyClass instance at 0x7fdd2537ea28>, 1, 2),{'y': 4, 'x': 3})

# class methods always get the class as the first argument

o.class_method
# <bound method classobj.class_method of <class __main__.MyClass at 0x7fdd2536a390>>

o.class_method()
# class_method((<class __main__.MyClass at 0x7fdd2536a390>,),{})

o.class_method(1,2,x=3,y=4)
# class_method((<class __main__.MyClass at 0x7fdd2536a390>, 1, 2),{'y': 4, 'x': 3})

# static methods have no arguments except the ones you pass in when you call them

o.static_method
# <function static_method at 0x7fdd25375848>

o.static_method()
# static_method((),{})

o.static_method(1,2,x=3,y=4)
# static_method((1, 2),{'y': 4, 'x': 3})

# properties are a way of implementing getters and setters. It's an error to explicitly call them
# "read only" attributes can be specified by creating a getter without a setter (as in some_other_property)

o.some_property
# calling some_property getter(<__main__.MyClass instance at 0x7fb2b70877e8>,(),{})
# 'properties are nice'

o.some_property()
# calling some_property getter(<__main__.MyClass instance at 0x7fb2b70877e8>,(),{})
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: 'str' object is not callable

o.some_other_property
# calling some_other_property getter(<__main__.MyClass instance at 0x7fb2b70877e8>,(),{})
# 'VERY nice'

# o.some_other_property()
# calling some_other_property getter(<__main__.MyClass instance at 0x7fb2b70877e8>,(),{})
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: 'str' object is not callable

o.some_property = "groovy"
# calling some_property setter(<__main__.MyClass object at 0x7fb2b7077890>,('groovy',),{})

o.some_property
# calling some_property getter(<__main__.MyClass object at 0x7fb2b7077890>,(),{})
# 'groovy'

o.some_other_property = "very groovy"
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: can't set attribute

o.some_other_property
# calling some_other_property getter(<__main__.MyClass object at 0x7fb2b7077890>,(),{})
# 'VERY nice'
```
### Question 10 Consider the following code, what will it output?
```python
class A(object):
    def go(self):
        print("go A go!")
    def stop(self):
        print("stop A stop!")
    def pause(self):
        raise Exception("Not Implemented")

class B(A):
    def go(self):
        super(B, self).go()
        print("go B go!")

class C(A):
    def go(self):
        super(C, self).go()
        print("go C go!")
    def stop(self):
        super(C, self).stop()
        print("stop C stop!")

class D(B,C):
    def go(self):
        super(D, self).go()
        print("go D go!")
    def stop(self):
        super(D, self).stop()
        print("stop D stop!")
    def pause(self):
        print("wait D wait!")

class E(B,C): pass

a = A()
b = B()
c = C()
d = D()
e = E()

# specify output from here onwards

a.go()
b.go()
c.go()
d.go()
e.go()

a.stop()
b.stop()
c.stop()
d.stop()
e.stop()

a.pause()
b.pause()
c.pause()
d.pause()
e.pause()
```
Answer
The output is specified in the comments in the segment below:
```python
a.go()
# go A go!

b.go()
# go A go!
# go B go!

c.go()
# go A go!
# go C go!
 
d.go()
# go A go!
# go C go!
# go B go!
# go D go!

e.go()
# go A go!
# go C go!
# go B go!

a.stop()
# stop A stop!

b.stop()
# stop A stop!

c.stop()
# stop A stop!
# stop C stop!

d.stop()
# stop A stop!
# stop C stop!
# stop D stop!

e.stop()
# stop A stop!
 
a.pause()
# ... Exception: Not Implemented

b.pause()
# ... Exception: Not Implemented

c.pause()
# ... Exception: Not Implemented

d.pause()
# wait D wait!

e.pause()
# ...Exception: Not Implemented
```
Why do we care?
Because OO programming is really, really important. Really. Answering this question shows your understanding of inheritance and the use of Python's super function. Most of the time the order of resolution doesn't matter. Sometimes it does, it depends on your application.
Question 11
Consider the following code, what will it output?
```python
class Node(object):
    def __init__(self,sName):
        self._lChildren = []
        self.sName = sName
    def __repr__(self):
        return "<Node '{}'>".format(self.sName)
    def append(self,*args,**kwargs):
        self._lChildren.append(*args,**kwargs)
    def print_all_1(self):
        print(self)
        for oChild in self._lChildren:
            oChild.print_all_1()
    def print_all_2(self):
        def gen(o):
            lAll = [o,]
            while lAll:
                oNext = lAll.pop(0)
                lAll.extend(oNext._lChildren)
                yield oNext
        for oNode in gen(self):
            print(oNode)

oRoot = Node("root")
oChild1 = Node("child1")
oChild2 = Node("child2")
oChild3 = Node("child3")
oChild4 = Node("child4")
oChild5 = Node("child5")
oChild6 = Node("child6")
oChild7 = Node("child7")
oChild8 = Node("child8")
oChild9 = Node("child9")
oChild10 = Node("child10")

oRoot.append(oChild1)
oRoot.append(oChild2)
oRoot.append(oChild3)
oChild1.append(oChild4)
oChild1.append(oChild5)
oChild2.append(oChild6)
oChild4.append(oChild7)
oChild3.append(oChild8)
oChild3.append(oChild9)
oChild6.append(oChild10)

# specify output from here onwards

oRoot.print_all_1()
oRoot.print_all_2()
```
Answer
oRoot.print_all_1() prints:
```python
<Node 'root'>
<Node 'child1'>
<Node 'child4'>
<Node 'child7'>
<Node 'child5'>
<Node 'child2'>
<Node 'child6'>
<Node 'child10'>
<Node 'child3'>
<Node 'child8'>
<Node 'child9'>
oRoot.print_all_2() prints:
<Node 'root'>
<Node 'child1'>
<Node 'child2'>
<Node 'child3'>
<Node 'child4'>
<Node 'child5'>
<Node 'child6'>
<Node 'child8'>
<Node 'child9'>
<Node 'child7'>
<Node 'child10'>
```
Why do we care?
Because composition and object construction is what objects are all about. Objects are composed of stuff and they need to be initialised somehow. This also ties up some stuff about recursion and use of generators.
Generators are great. You could have achieved similar functionality to print_all_2by just constructing a big long list and then printing it's contents. One of the nice things about generators is that they don't need to take up much space in memory.
It is also worth pointing out that print_all_1 traverses the tree in a depth-first manner, while print_all_2 is width-first. Make sure you understand those terms. Sometimes one kind of traversal is more appropriate than the other. But that depends very much on your application.
Question 12
Describe Python's garbage collection mechanism in brief.
Answer
A lot can be said here. There are a few main points that you should mention:
- Python maintains a count of the number of references to each object in memory. If a reference count goes to zero then the associated object is no longer live and the memory allocated to that object can be freed up for something else
- occasionally things called "reference cycles" happen. The garbage collector periodically looks for these and cleans them up. An example would be if you have two objects o1 and o2 such that o1.x == o2 and o2.x == o1. If o1 and o2 are not referenced by anything else then they shouldn't be live. But each of them has a reference count of 1.
- Certain heuristics are used to speed up garbage collection. For example, recently created objects are more likely to be dead. As objects are created, the garbage collector assigns them to generations. Each object gets one generation, and younger generations are dealt with first.
This explanation is CPython specific.
Question 13
Place the following functions below in order of their efficiency. They all take in a list of numbers between 0 and 1. The list can be quite long. An example input list would be [random.random() for i in range(100000)]. How would you prove that your answer is correct?
```python
def f1(lIn):
    l1 = sorted(lIn)
    l2 = [i for i in l1 if i<0.5]
    return [i*i for i in l2]

def f2(lIn):
    l1 = [i for i in lIn if i<0.5]
    l2 = sorted(l1)
    return [i*i for i in l2]

def f3(lIn):
    l1 = [i*i for i in lIn]
    l2 = sorted(l1)
    return [i for i in l1 if i<(0.5*0.5)]
```
Answer
Most to least efficient: f2, f1, f3. To prove that this is the case, you would want to profile your code. Python has a lovely profiling package that should do the trick.
```python
import cProfile
lIn = [random.random() for i in range(100000)]
cProfile.run('f1(lIn)')
cProfile.run('f2(lIn)')
cProfile.run('f3(lIn)')
```
For completion's sake, here is what the above profile outputs:
```python
>>> cProfile.run('f1(lIn)')
         4 function calls in 0.045 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.009    0.009    0.044    0.044 <stdin>:1(f1)
        1    0.001    0.001    0.045    0.045 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.035    0.035    0.035    0.035 {sorted}


>>> cProfile.run('f2(lIn)')
         4 function calls in 0.024 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.008    0.008    0.023    0.023 <stdin>:1(f2)
        1    0.001    0.001    0.024    0.024 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.016    0.016    0.016    0.016 {sorted}


>>> cProfile.run('f3(lIn)')
         4 function calls in 0.055 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.016    0.016    0.054    0.054 <stdin>:1(f3)
        1    0.001    0.001    0.055    0.055 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.038    0.038    0.038    0.038 {sorted}
```
Why Care?
Locating and avoiding bottlenecks is often pretty worthwhile. A lot of coding for efficiency comes down to common sense - in the example above it's obviously quicker to sort a list if it's a smaller list, so if you have the choice of filtering before a sort it's often a good idea. The less obvious stuff can still be located through use of the proper tools. It's good to know about these tools.
Question 14
Something you failed at?
Wrong answer
I never fail!
Why This Is Important:
Shows that you are capable of admitting errors, taking responsibility for your mistakes, and learning from your mistakes. All of these things are pretty darn important if you are going to be useful. If you are actually perfect then too bad, you might need to get creative here.
### Question 15
Do you have any personal projects?
Really?
This shows that you are willing to do more than the bare minimum in terms of keeping your skillset up to date. If you work on personal projects and code outside of the workplace then employers are more likely to see you as an asset that will grow. Even if they don't ask this question I find it's useful to broach the subject.


### Q.1. What are the key features of Python?
If it makes for an introductory language to programming, Python must mean something. These are its qualities:
1.	Interpreted
2.	Dynamically-typed
3.	Object-oriented
4.	Concise and simple
5.	Free
6.	Has a large community
### Q.2. Differentiate between deep and shallow copy.
A deep copy copies an object into another. This means that if you make a change to a copy of an object, it won’t affect the original object. In Python, we use the function deepcopy() for this, and we import the module copy. We use it like:
```python
1.	>>> import copy
2.	>>> b=copy.deepcopy(a)

```

![picture alt]( https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/Python-deep-copy.jpg)

A shallow copy, however, copies one object’s reference to another. So, if we make a change in the copy, it will affect the original object. For this, we have the function copy(). We use it like:
```python
1.	>>> b=copy.copy(a)

```
![picture alt]( https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/Python-shallow-copy.jpg)

### Q.3. Differentiate between lists and tuples.
The major difference is that a list is mutable, but a tuple is immutable. Examples:
```python
1.	>>> mylist=[1,3,3]
2.	>>> mylist[1]=2
3.	>>> mytuple=(1,3,3)
4.	>>> mytuple[1]=2
5.	Traceback (most recent call last):
6.	File "<pyshell#97>", line 1, in <module>
7.	mytuple[1]=2
```

TypeError: ‘tuple’ object does not support item assignment
### Q.4. Explain the ternary operator in Python.
Unlike C++, we don’t have ?: in Python, but we have this:
[on true] if [expression] else [on false]
If the expression is True, the statement under [on true] is executed. Else, that under [on false] is executed.
Below is how you would use it:
```python

1.	>>> a,b=2,3
2.	>>> min=a if a<b else b
3.	>>> min


2
1.	>>> print("Hi") if a<b else print("Bye")
Hi
```

### Q.5. How is multithreading achieved in Python?
A thread is a lightweight process, and multithreading allows us to execute multiple threads at once. As you know, Python is a multithreaded language. It has a multi-threading package.
The GIL (Global Interpreter Lock) ensures that a single thread executes at a time. A thread holds the GIL and does a little work before passing it on to the next thread. This makes for an illusion of parallel execution. But in reality, it is just threads taking turns at the CPU. Of course, all the passing around adds overhead to the execution.
### Q.6. Explain inheritance.
When one class inherits from another, it is said to be the child/derived/sub class inheriting from the parent/base/super class. It inherits/gains all members (attributes and methods).
Inheritance lets us reuse our code, and also makes it easier to create and maintain applications. Python supports the following kinds of inheritance:
1.	Single Inheritance- A class inherits from a single base class.
2.	Multiple Inheritance- A class inherits from multiple base classes.
3.	Multilevel Inheritance- A class inherits from a base class, which, in turn, inherits from another base class.
4.	Hierarchical Inheritance- Multiple classes inherit from a single base class.
5.	Hybrid Inheritance- Hybrid inheritance is a combination of two or more types of inheritance.
For more on inheritance, refer to Python Inheritance.
### Q.7. What is Flask?
Flask, as we’ve previously discussed, is a web microframework for Python. It is based on the ‘Werkzeug, Jinja 2 and good intentions’ BSD license. Two of its dependencies are Werkzeug and Jinja2. This means it has around no dependencies on external libraries. Due to this, we can call it a light framework.
A session uses a signed cookie to allow for the user to look at and modify session contents. It will remember information from one request to another. However, to modify a session, the user must have the secret key Flask.secret_key.
We will discuss Flask in greater detail in a further lesson.
### Q.8. How is memory managed in Python?
Python has a private heap space to hold all objects and data structures. Being programmers, we cannot access it; it is the interpreter that manages it. But with the core API, we can access some tools. The Python memory manager controls the allocation.
Additionally, an inbuilt garbage collector recycles all unused memory so it can make it available to the heap space.
### Q.9. Explain help() and dir() functions in Python.
The help() function displays the documentation string and help for its argument.
```python

1.	>>> import copy
2.	>>> help(copy.copy)
```
Help on function copy in module copy:
 
copy(x)
Shallow copy operation on arbitrary Python objects.
See the module’s __doc__ string for more info.
The dir() function displays all the members of an object(any kind).
```python
1.	>>> dir(copy.copy)
```
[‘__annotations__’, ‘__call__’, ‘__class__’, ‘__closure__’, ‘__code__’, ‘__defaults__’, ‘__delattr__’, ‘__dict__’, ‘__dir__’, ‘__doc__’, ‘__eq__’, ‘__format__’, ‘__ge__’, ‘__get__’, ‘__getattribute__’, ‘__globals__’, ‘__gt__’, ‘__hash__’, ‘__init__’, ‘__init_subclass__’, ‘__kwdefaults__’, ‘__le__’, ‘__lt__’, ‘__module__’, ‘__name__’, ‘__ne__’, ‘__new__’, ‘__qualname__’, ‘__reduce__’, ‘__reduce_ex__’, ‘__repr__’, ‘__setattr__’, ‘__sizeof__’, ‘__str__’, ‘__subclasshook__’]
### Q.10. Whenever you exit Python, is all memory de-allocated?
The answer here is no. The modules with circular references to other objects, or to objects referenced from global namespaces, aren’t always freed on exiting Python.
Plus, it is impossible to de-allocate portions of memory reserved by the C library.
### Q.11. What is monkey patching?
Dynamically modifying a class or module at run-time.
```python
1.	>>> class A:
2.	def func(self):
3.	print("Hi")
4.	>>> def monkey(self):
5.	print "Hi, monkey"
6.	>>> m.A.func = monkey
7.	>>> a = m.A()
8.	>>> a.func()
Hi, monkey
```
### Q.12. What is a dictionary in Python?
A dictionary is something I have never seen in other languages like C++ or Java. It holds key-value pairs.
```python
1.	>>> roots={25:5,16:4,9:3,4:2,1:1}
2.	>>> type(roots)
3.	<class 'dict'>
4.	>>> roots[9]
3
```

A dictionary is mutable, and we can also use a comprehension to create it.
```python
1.	>>> roots={x**2:x for x in range(5,0,-1)}
2.	>>> roots
{25: 5, 16: 4, 9: 3, 4: 2, 1: 1}
```

### Q.13. What do you mean by *args and **kwargs?
In cases when we don’t know how many arguments will be passed to a function, like when we want to pass a list or a tuple of values, we use *args.
```python
1.	>>> def func(*args):
2.	for i in args:
3.	print(i) 
4.	>>> func(3,2,1,4,7)
3
2
1
4
7
```
**kwargs takes keyword arguments when we don’t know how many there will be.
```python
1.	>>> def func(**kwargs):
2.	for i in kwargs:
3.	print(i,kwargs[i])
4.	>>> func(a=1,b=2,c=7)
a.1
b.2
c.7
```
The words args and kwargs are convention, and we can use anything in their place.
Any doubt yet in Basic Python Interview Questions and answers for Freshers? Please ask in Comments.
### Q.14. Write Python logic to count the number of capital letters in a file.
```python
1.	>>> import os
2.	>>> os.chdir('C:\\Users\\lifei\\Desktop')
3.	>>> with open('Today.txt') as today:
4.	count=0
5.	for i in today.read():
6.	if i.isupper():
7.	count+=1
8.	print(count)
26
```
### Q.15. What are negative indices?
Let’s take a list for this.
```python

1.	>>> mylist=[0,1,2,3,4,5,6,7,8]

```
A negative index, unlike a positive one, begins searching from the right.
```python

1.	>>> mylist[-3]
6
```
This also helps with slicing from the back:
```python

1.	>>> mylist[-6:-1]
[3, 4, 5, 6, 7]
```

### Q.16. How would you randomize the contents of a list in-place?
For this, we’ll import the function shuffle() from the module random.
```python
1.	>>> from random import shuffle
2.	>>> shuffle(mylist)
3.	>>> mylist
[3, 4, 8, 0, 5, 7, 6, 2, 1]
```
### Q.17. Explain join() and split() in Python.
join() lets us join characters from a string together by a character we specify.
```python
1.	>>> ','.join('12345')
‘1,2,3,4,5’
```
split() lets us split a string around the character we specify.
```python
1.	>>> '1,2,3,4,5'.split(',')
[‘1’, ‘2’, ‘3’, ‘4’, ‘5’]
```
### Q.18. Is Python case-sensitive?
A language is case-sensitive if it distinguishes between identifiers like myname and Myname. In other words, it cares about case- lowercase or uppercase. Let’s try this with Python.
```python
1.	>>> myname='Ayushi'
2.	>>> Myname
3.	Traceback (most recent call last):
4.	File "<pyshell#3>", line 1, in <module>
Myname
NameError: name ‘Myname’ is not defined
```
As you can see, this raised a NameError. This means that Python is indeed case-sensitive.
### Q.19. How long can an identifier be in Python?
In Python, an identifier can be of any length. Apart from that, there are certain rules we must follow to name one:
1.	It can only begin with an underscore or a character from A-Z or a-z.
2.	The rest of it can contain anything from the following: A-Z/a-z/_/0-9.
3.	Python is case-sensitive, as we discussed in the previous question.
### Q.20. How do you remove the leading whitespace in a string?
Leading whitespace in a string is the whitespace in a string before the first non-whitespace character. To remove it from a string, we use the method lstrip().
```python
1.	>>> ' Ayushi '.lstrip()
‘Ayushi   ‘
```
As you can see, this string had both leading and trailing whitespaces. lstrip() stripped the string of the leading whitespace. If we want to strip the trailing whitespace instead, we use rstrip().
```python
1.	>>> ' Ayushi '.rstrip()
‘   Ayushi’
```
### Q.21. How would you convert a string into lowercase?
We use the lower() method for this.
```python
1.	>>> 'AyuShi'.lower()
‘ayushi’
```
To convert it into uppercase, then, we use upper().
```python
1.	>>> 'AyuShi'.upper()
‘AYUSHI’
```
Also, to check if a string is in all uppercase or all lowercase, we use the methods isupper() and islower().
```python
1.	>>> 'AyuShi'.isupper()
False
1.	>>> 'AYUSHI'.isupper()
True
1.	>>> 'ayushi'.islower()
True
1.	>>> '@yu$hi'.islower()
True
1.	>>> '@YU$HI'.isupper()
True
```
So, characters like @ and $ will suffice for both cases.
Also, istitle() will tell us if a string is in title case.
```python
1.	>>> 'The Corpse Bride'.istitle()
True
```
### Q.22. What is the pass statement in Python?
There may be times in our code when we haven’t decided what to do yet, but we must type something for it to be syntactically correct. In such a case, we use the pass statement.
```python
1.	>>> def func(*args):
2.	pass 
3.	>>>
```
Similarly, the break statement breaks out of a loop.
```python
1.	>>> for i in range(7):
2.	if i==3: break
3.	print(i)
0
1
2
```
Finally, the continue statement skips to the next iteration.
```python
1.	>>> for i in range(7):
2.	if i==3: continue
3.	print(i)
0
1
2
4
5
6
```
### Q.23. What is a closure in Python?
A closure is said to occur when a nested function references a value in its enclosing scope. The whole point here is that it remembers the value.
```python
1.	>>> def A(x):
2.	def B():
3.	print(x)
4.	return B
5.	>>> A(7)()
7
```
### Q.24. Explain the //, %, and ** operators in Python.
The // operator performs floor division. It will return the integer part of the result on division.
1.	>>> 7//2
3
Normal division would return 3.5 here.
Similarly, ** performs exponentiation. a**b returns the value of a raised to the power b.
1.	>>> 2**10
1024
Finally, % is for modulus. This gives us the value left after the highest achievable division.
1.	>>> 13%7
6
1.	>>> 3.5%1.5
0.5
Any Doubt yet in Advanced Python Interview Questions and Answers for Experienced? Please Comment.
### Q.24. How many kinds of operators do we have in Python? Explain arithmetic operators.
This type of Python Interview Questions and Answers can decide your knowledge in Python. Answer the Python Interview Questions with some good Examples.
Here in Python, we have 7 kinds of operators: arithmetic, relational, assignment, logical, membership, identity, and bitwise.
We have seven arithmetic operators. These allow us to perform arithmetic operations on values:
1.	Addition (+) This adds two values.
1.	>>> 7+8
15
2.	Subtraction (-) This subtracts he second value from the first.
1.	>>> 7-8
-1
3.	Multiplication (*) This multiplies two numbers.
1.	>>> 7*8
56
4.	Division (/) This divides the first value by the second.
1.	>>> 7/8
0.875
1.	>>> 1/1
1.0
For floor division, modulus, and exponentiation, refer to the previous question.
### Q.25. Explain relational operators in Python.
Relational operators compare values.
1.	Less than (<) If the value on the left is lesser, it returns True.
1.	>>> 'hi'<'Hi'
False
2.	Greater than (>) If the value on the left is greater, it returns True.
1.	>>> 1.1+2.2>3.3
True
This is because of the flawed floating-point arithmetic in Python, due to hardware dependencies.
3.	Less than or equal to (<=) If the value on the left is lesser than or equal to, it returns True.
1.	>>> 3.0<=3
True
4.	Greater than or equal to (>=) If the value on the left is greater than or equal to, it returns True.
1.	>>> True>=False
True
5.	Equal to (==) If the two values are equal, it returns True.
1.	>>> {1,3,2,2}=={1,2,3}
True
6.	Not equal to (!=) If the two values are unequal, it returns True.
1.	>>> True!=0.1
True
1.	>>> False!=0.1
True
### Q.26. What are assignment operators in Python?
This one is an Important Interview question in Python Interview.
We can combine all arithmetic operators with the assignment symbol.
1.	>>> a=7
2.	>>> a+=1
3.	>>> a
8
1.	>>> a-=1
2.	>>> a
7
1.	>>> a*=2
2.	>>> a
14
1.	>>> a/=2
2.	>>> a
7.0
1.	>>> a**=2
2.	>>> a
49.0
1.	>>> a//=3
2.	>>> a
16.0
1.	>>> a%=4
2.	>>> a
0.0
### Q.27. Explain logical operators in Python.
We have three logical operators- and, or, not.
1.	>>> False and True
False
1.	>>> 7<7 or True
True
1.	>>> not 2==2
False
### Q.28. What are membership operators?
With the operators ‘in’ and ‘not in’, we can confirm if a value is a member in another.
1.	>>> 'me' in 'disappointment'
True
1.	>>> 'us' not in 'disappointment'
True
### Q.29. Explain identity operators in Python.
This is one of the very commonly asked Python Interview Questions and answer it with examples.
The operators ‘is’ and ‘is not’ tell us if two values have the same identity.
1.	>>> 10 is '10'
False
1.	>>> True is not False
True
### Q.30. Finally, tell us about bitwise operators in Python.
These operate on values bit by bit.
1.	AND (&) This performs & on each bit pair.
1.	>>> 0b110 & 0b010
2
2.	OR (|) This performs | on each bit pair.
1.	>>> 3|2
3
3.	XOR (^) This performs an exclusive-OR operation on each bit pair.
1.	>>> 3^2
1
4.	Binary One’s Complement (~) This returns the one’s complement of a value.
1.	>>> ~2
-3
5.	Binary Left-Shift (<<) This shifts the bits to the left by the specified amount.
1.	>>> 1<<2
4
Here, 001 was shifted to the left by two places to get 100, which is binary for 4.
6.	Binary Right-Shift (>>)
1.	>>> 4>>2
1
For more insight on operators, refer to Operators in Python.
### Q.31. How would you work with numbers other than those in the decimal number system?
With Python, it is possible to type numbers in binary, octal, and hexadecimal.
1.	Binary numbers are made of 0 and 1. To type in binary, we use the prefix 0b or 0B.
1.	>>> int(0b1010)
10
To convert a number into its binary form, we use bin().
1.	>>> bin(0xf)
‘0b1111’
2.	Octal numbers may have digits from 0 to 7. We use the prefix 0o or 0O.
1.	>>> oct(8)
‘0o10’
3.	Hexadecimal numbers may have digits from 0 to 15. We use the prefix 0x or 0X.
1.	>>> hex(16)
‘0x10’
1.	>>> hex(15)
‘0xf’
### Q.32. How do you get a list of all the keys in a dictionary?
Be specific in these type of Python Interview Questions and Answers.
For this, we use the function keys().
1.	>>> mydict={'a':1,'b':2,'c':3,'e':5}
2.	>>> mydict.keys()
3.	dict_keys(['a', 'b', 'c', 'e'])
### Q.33. Why are identifier names with a leading underscore disparaged?
Since Python does not have a concept of private variables, it is a convention to use leading underscores to declare a variable private. This is why we mustn’t do that to variables we do not want to make private.
### Q.34. How can you declare multiple assignments in one statement?
There are two ways to do this:
1.	>>> a,b,c=3,4,5 #This assigns 3, 4, and 5 to a, b, and c respectively
2.	>>> a=b=c=3 #This assigns 3 to a, b, and c
### Q.35. What is tuple unpacking?
First, let’s discuss tuple packing. It is a way to pack a set of values into a tuple.
1.	>>> mytuple=3,4,5
2.	>>> mytuple
(3, 4, 5)
This packs 3, 4, and 5 into mytuple.
Now, we will unpack the values from the tuple into variables x, y, and z.
1.	>>> x,y,z=mytuple
2.	>>> x+y+z
12
These were the Advanced Python Interview Questions and Answers for Experiences. Freshers may Also Refer the Python Interview Questions for advanced knowledge.

## intellipaat

### Q2. How are the functions help() and dir() different?
These are the two functions that are accessible from the Python Interpreter. These two functions are used for viewing a consolidated dump of built-in functions.

help() – it will display the documentation string. It is used to see the help related to modules, keywords, attributes, etc.
To view the help related to string datatype, just execute a statement help(str) – it will display the documentation for ‘str, module. ◦ Eg: >>>help(str) or >>>help() – it will open the prompt for help as help>
to view the help for a module, help> module module name Inorder to view the documentation of ‘str’ at the help>, type help>modules str
to view the help for a keyword, topics, you just need to type, help> “keywords python- keyword” and “topics list”
dir() – will display the defined symbols. Eg: >>>dir(str) – will only display the defined symbols.
Go through this Python tutorial to learn more about Python Functions.

### Q3. Which command do you use to exit help window or help command prompt?
quit
When you type quit at the help’s command prompt, python shell prompt will appear by closing the help window automatically.

Check this insightful tutorial to learn more about Python Program & its Execution.

 

### Q4. Does the functions help() and dir() list the names of all the built_in functions and variables? If no, how would you list them?
No. Built-in functions such as max(), min(), filter(), map(), etc is not apparent immediately as they are
available as part of standard module.To view them, we can pass the module ” builtins ” as an argument to “dir()”. It will display the
built-in functions, exceptions, and other objects as a list.>>>dir(__builtins )
[‘ArithmeticError’, ‘AssertionError’, ‘AttributeError’, ……… ]

### Q5. Explain how Python does Compile-time and Run-time code checking?
Python performs some amount of compile-time checking, but most of the checks such as type, name, etc are postponed until code execution. Consequently, if the Python code references a user -defined function that does not exist, the code will compile successfully. In fact, the code will fail with an exception only when the code execution path references the function which does not exists.

Read this blog to learn how to deploy automated coding with Python.

### Q6. Whenever Python exists Why does all the memory is not de-allocated / freed when Python exits?
Whenever Python exits, especially those python modules which are having circular references to other objects or the objects that are referenced from the global namespaces are not always de – allocated/freed/uncollectable.
It is impossible to deallocate those portions of memory that are reserved by the C library.
On exit, because of having its own efficient clean up mechanism, Python would try to deallocate/
destroy every object.

### Q7. Explain Python's zip() function.?
zip() function- it will take multiple lists say list1, list2, etc and transform them into a single list of tuples by taking the corresponding elements of the lists that are passed as parameters. Eg:

list1 = ['A',
'B','C'] and list2 = [10,20,30].
zip(list1, list2) # results in a list of tuples say [('A',10),('B',20),('C',30)]
whenever the given lists are of different lengths, zip stops generating tuples when the first list ends.

### Q8. Explain Python's pass by references Vs pass by value . (or) Explain about Python's parameter passing mechanism?
In Python, by default, all the parameters (arguments) are passed “by reference” to the functions. Thus, if you change the value of the parameter within a function, the change is reflected in the calling function.We can even observe the pass “by value” kind of a behaviour whenever we pass the arguments to functions that are of type say numbers, strings, tuples. This is because of the immutable nature of them.

### Q9. As Everything in Python is an Object, Explain the characteristics of Python's Objects.
As Python’s Objects are instances of classes, they are created at the time of instantiation. Eg: object-name = class-name(arguments)
one or more variables can reference the same object in Python
Every object holds unique id and it can be obtained by using id() method. Eg: id(obj-name) will return unique id of the given object.
every object can be either mutable or immutable based on the type of data they hold.
 Whenever an object is not being used in the code, it gets destroyed automatically garbage collected or destroyed
 contents of objects can be converted into string representation using a method
Go through this Python Video to get clear understanding of Python.

### Q10. Explain how to overload constructors or methods in Python.
Python’s constructor – _init__ () is a first method of a class. Whenever we try to instantiate a object __init__() is automatically invoked by python to initialize members of an object.

### Q11. Which statement of Python is used whenever a statement is required syntactically but the program needs no action?
Pass – is no-operation / action statement in Python
If we want to load a module or open a file, and even if the requested module/file does not exist, we want to continue with other tasks. In such a scenario, use try-except block with pass statement in the except block.
Eg:

 try:import mymodulemyfile = open(“C:\myfile.csv”)except:pass

### Q12. What is Web Scraping? How do you achieve it in Python?
Web Scrapping is a way of extracting the large amounts of information which is available on the web sites and saving it onto the local machine or onto the database tables.
In order to scrap the web:load the web page which is interesting to you. To load the web page, use “requests” module.
parse HTML from the web page to find the interesting information.Python has few modules for scraping the web. They are urllib2, scrapy, pyquery, BeautifulSoap, etc.

### Q13. What is a Python module?
A module is a Python script that generally contains import statements, functions, classes and variable definitions, and Python runnable code and it “lives” file with a ‘.py’ extension. zip files and DLL files can also be modules.Inside the module, you can refer to the module name as a string that is stored in the global variable name .
A module can be imported by other modules in one of the two ways. They are

 import
from module-name import
Check this tutorial to learn more about Python modules.

### Q14. Name the File-related modules in Python?
Python provides libraries / modules with functions that enable you to manipulate text files and binary files on file system. Using them you can create files, update their contents, copy, and delete files. The libraries are : os, os.path, and shutil.
Here, os and os.path – modules include functions for accessing the filesystem
shutil – module enables you to copy and delete the files.

### Q15. Explain the use of with statement?
In python generally “with” statement is used to open a file, process the data present in the file, and also to close the file without calling a close() method. “with” statement makes the exception handling simpler by providing cleanup activities.
General form of with:
with open(“file name”, “mode”) as file-var:
processing statements
note: no need to close the file by calling close() upon file-var.close()

### Q16. Explain all the file processing modes supported by Python ?
Python allows you to open files in one of the three modes. They are:
read-only mode, write-only mode, read-write mode, and append mode by specifying the flags “r”, “w”, “rw”, “a” respectively.
A text file can be opened in any one of the above said modes by specifying the option “t” along with
“r”, “w”, “rw”, and “a”, so that the preceding modes become “rt”, “wt”, “rwt”, and “at”.A binary file can be opened in any one of the above said modes by specifying the option “b” along with “r”, “w”, “rw”, and “a” so that the preceding modes become “rb”, “wb”, “rwb”, “ab”.

### Q17. Explain how to redirect the output of a python script from standout(ie., monitor) on to a file ?
They are two possible ways of redirecting the output from standout to a file.

 Open an output file in “write” mode and the print the contents in to that file, using sys.stdout attribute.
import sys
filename = “outputfile” sys.stdout = open() print “testing”
you can create a python script say .py file with the contents, say print “testing” and then redirect it to the output file while executing it at the command prompt.
Eg: redirect_output.py has the following code:
print “Testing”
execution: python redirect_output.py > outputfile.
Go through this Python tutorial to get better understanding of  Python Input & Output.

### Q18. Explain the shortest way to open a text file and display its contents.?
The shortest way to open a text file is by using “with” command as follows:

with open("file-name", "r") as fp:
fileData = fp.read()
#to print the contents of the file print(fileData)

### Q19. How do you create a dictionary which can preserve the order of pairs?
We know that regular Python dictionaries iterate over <key, value> pairs in an arbitrary order, hence they do not preserve the insertion order of <key, value> pairs.
Python 2.7. introduced a new “OrderDict” class in the “collections” module and it provides the same interface like the general dictionaries but it traverse through keys and values in an ordered manner depending on when a key was first inserted.
Eg:

from collections import OrderedDict
d = OrderDict([('Company-id':1),('Company-Name':'Intellipaat')])
d.items() # displays the output as: [('Company-id':1),('Company-Name':'Intellipaat')]

### Q20. When does a dictionary is used instead of a list?
Dictionaries – are best suited when the data is labelled, i.e., the data is a record with field names.
lists – are better option to store collections of un-labelled items say all the files and sub directories in a folder. List comprehension is used to construct lists in a natural way.
Generally Search operation on dictionary object is faster than searching a list object.

### Q21. What is the use of enumerate() in Python?
Using enumerate() function you can iterate through the sequence and retrieve the index position and its corresponding value at the same time.
>>> for i,v in enumerate([‘Python’,’Java’,’C++’]):
print(i,v)
0 Python
1 Java
2 C++

### Q22. How many kinds of sequences are supported by Python? What are they?
Python supports 7 sequence types. They are str, list, tuple, unicode, bytearray, xrange, and buffer. where xrange is deprecated in python 3.5.X.

### Q23. How do you perform pattern matching in Python? Explain
Regular Expressions/REs/ regexes enable us to specify expressions that can match specific “parts” of a given string. For instance, we can define a regular expression to match a single character or a digit, a telephone number, or an email address, etc.The Python’s “re” module provides regular expression patterns and was introduce from later versions of Python 2.5. “re” module is providing methods for search text strings, or replacing text strings along with methods for splitting text strings based on the pattern defined.

### Q24. Name few methods for matching and searching the occurrences of a pattern in a given text String ?
There are 4 different methods in “re” module to perform pattern matching. They are:
match() – matches the pattern only to the beginning of the String. search() – scan the string and look for a location the pattern matches findall() – finds all the occurrences of match and return them as a list
finditer() – finds all the occurrences of match and return them as an iterator.

### Q25. Explain split(), sub(), subn() methods of
To modify the strings, Python’s “re” module is providing 3 methods. They are:
split() – uses a regex pattern to “split” a given string into a list.
sub() – finds all substrings where the regex pattern matches and then replace them with a different string
subn() – it is similar to sub() and also returns the new string along with the no. of
replacements.

### Q26. How to display the contents of text file in reverse order?
convert the given file into a list.
 reverse the list by using reversed()
Eg: for line in reversed(list(open(“file-name”,”r”))):
print(line)
### Q27. What is JSON? How would convert JSON data into Python data?
JSON – stands for JavaScript Object Notation. It is a popular data format for storing data in NoSQL
databases. Generally JSON is built on 2 structures.

 A collection of <name, value> pairs.
 An ordered list of values.
As Python supports JSON parsers, JSON-based data is actually represented as a dictionary in Python. You can convert json data into python using load() of json module.

### Q28. Name few Python modules for Statistical, Numerical and scientific computations ?
numPy – this module provides an array/matrix type, and it is useful for doing computations on arrays. scipy – this module provides methods for doing numeric integrals, solving differential equations, etc pylab – is a module for generating and saving plots
matplotlib – used for managing data and generating plots.

### Q29. What is TkInter?
TkInter is Python library. It is a toolkit for GUI development. It provides support for various GUI tools or widgets (such as buttons, labels, text boxes, radio buttons, etc) that are used in GUI applications. The common attributes of them include Dimensions, Colors, Fonts, Cursors, etc.

### Q30. Name and explain the three magic methods of Python that are used in the construction and initialization of custom Objects.
The 3 magic methods of Python that are used in the construction and initialization of custom Objects are: init__, new , and del__.
new – this method can be considered as a “constructor”. It is invoked to create an instance of a class with the statement say, myObj = MyClass()
init__ — It is an “initializer”/ “constructor” method. It is invoked whenever any arguments are passed at the time of creating an object. myObj = MyClass(‘Pizza’,25)
del- this method is a “destructor” of the class. Whenever an object is deleted,
invocation of del__ takes place and it defines behaviour during the garbage collection. Note: new , del are rarely used explicitly.

### Q31. Is Python object oriented? what is object oriented programming?
Yes. Python is Object Oriented Programming language. OOP is the programming paradigm based on classes and instances of those classes called objects. The features of OOP are:
Encapsulation, Data Abstraction, Inheritance, Polymorphism.

### Q32. What is a Class? How do you create it in Python?
A class is a blue print/ template of code /collection of objects that has same set of attributes and behaviour. To create a class use the keyword class followed by class name beginning with an uppercase letter. For example, a person belongs to class called Person class and can have the attributes (say first-name and last-name) and behaviours / methods (say showFullName()). A Person class can be defined as:

class Person():
#method
def inputName(self,fname,lname): self.fname=fname self.lastname=lastname
#method
def showFullName() (self):
print(self.fname+" "+self.lname)person1 = Person() #object instantiation person1.inputName("Ratan","Tata") #calling a method inputName person1. showFullName() #calling a method showFullName()
Note: whenever you define a method inside a class, the first argument to the method must be self (where self – is a pointer to the class instance). self must be passed as an argument to the method, though the method does not take any arguments.

### Q33. What are Exception Handling? How do you achieve it in Python?
Exception Handling prevents the codes and scripts from breaking on receipt of an error at run -time might be at the time doing I/O, due to syntax errors, data types doesn’t match. Generally it can be used for handling user inputs.
The keywords that are used to handle exceptions in Python are:
try – it will try to execute the code that belongs to it. May be it used anywhere that keyboard input is required.
except – catches all errors or can catch a specific error. It is used after the try block.x = 10 + ‘Python’ #TypeError: unsupported operand type(s) …. try:
x = 10 + ‘Python’
except:
print(“incompatible operand types to perform sum”)
raise – force an error to occur
o raise TypeError(“dissimilar data types”)
finally – it is an optional clause and in this block cleanup code is written here following “try” and “except”.

### Q34. Explain Inheritance in Python with an example.
Inheritance allows One class to gain all the members(say attributes and methods) of another class. Inheritance provides code reusability, makes it easier to create and maintain an application. They are different types of inheritance supported by Python. They are: single, multi-level, hierarchical and multiple inheritance. The class from which we are inheriting is called super-class and the class that is inherited is called a derived / child class.
Single Inheritance – where a derived class acquires the members of a single super class.
multi-level inheritance – a derived class d1 in inherited from base class base1, and d2 is inherited from base2.
hierarchical inheritance – from one base class you can inherit any number of child classes
multiple inheritance – a derived class is inherited from more than one base class.
ex:

class ParentClass:
v1 = "from ParentClass - v1"
v2 = "from ParentClass - v2"class ChildClass(ParentClass):
passc = ChildClass() print(c.v1) print(c.v2)

### Q35. What is multithreading? Give an example.
It means running several different programs at the same time concurrently by invoking multiple threads. Multiple threads within a process refer the data space with main thread and they can communicate with each other to share information more easily.Threads are light-weight processes and have less memory overhead. Threads can be used just for quick task like calculating results and also running other processes in the background while the main program is running.

### Q36. How instance variables are different from class variables?
Instance variables: are the variables in an object that have values that are local to that object. Two objects of the same class maintain distinct values for their variables. These variables are accessed with “object-name.instancevariable-name”.
class variables: these are the variables of class. All the objects of the same class will share value of “Class variables. They are accessed with their class name alone as “class- name.classvariable-name”. If you change the value of a class variable in one object, its new value is visible among all other objects of the same class. In the Java world, a variable that is declared as static is a class variable.

### Q37. Explain different ways to trigger / raise exceptions in your python script ?
The following are the two possible ways by which you can trigger an exception in your Python script. They are:

raise — it is used to manually raise an exception general-form:
raise exception-name (“message to be conveyed”)
Eg: >>> voting_age = 15
>>> if voting_age < 18: raise ValueError(“voting age should be atleast 18 and above”) output: ValueError: voting age should be atleast 18 and above 2. assert statement assert statements are used to tell your program to test that condition attached to assert keyword, and trigger an exception whenever the condition becomes false. Eg: >>> a = -10
>>> assert a > 0 #to raise an exception whenever a is a negative number output: AssertionError
Another way of raising and exception can be done by making a programming mistake, but that’s not
usually a good way of triggering an exception.

### Q38. How is Inheritance and Overriding methods are related?
If class A is a sub class of class B, then everything in B is accessible in /by class A. In addition, class A can define methods that are unavailable in B, and also it is able to override methods in B. For Instance, If class B and class A both contain a method called func(), then func() in class B can override func() in class A. Similarly, a method of class A can call another method defined in A that can invoke a method of B that overrides it.

### Q39. Which methods of Python are used to determine the type of instance and inheritance?
Python has 2 built-in functions that work with inheritance:
isinstance() – this method checks the type of instance.

for eg, isinstance(myObj, int) – returns True only when “myObj. class ” is “int”.
issubclass() – this method checks class inheritance

for eg: issubclass(bool, int) – returns True because “bool” is a subclass of “int”.

issubclass(unicode, str) – returns False because “unicode” is not a subclass of “str”.

### Q40. In the case of Multiple inheritance, if a child class C is derived from two base classes say A and B as: class C(A, B): -- which parent class's method will be invoked by the interpreter whenever object of class C calls a method func() that is existing in both the parent classes say A and B and does not exist in class C as
since class C does not contain the definition of the method func(), they Python searches for the func() in parent classes. Since the search is performed in a left-to-right fashion, Python executes the method func() present in class A and not the func() method in B.

### Q41. Does Python supports interfaces like in Java? Discuss.
Python does not provide interfaces like in Java. Abstract Base Class (ABC) and its feature are provided by the Python’s “abc” module. Abstract Base Class is a mechanism for specifying what methods must be implemented by its implementation subclasses. The use of ABC’c provides a sort of “understanding” about methods and their expected behaviour. This module was made available from Python 2.7 version onwards.

### Q42. What are Accessors, mutators, @property?
Accessors and mutators are often called getters and setters in languages like “Java”. For example, if x is a property of a user-defined class, then the class would have methods called setX() and getX(). Python has an @property “decorator” that allows you to ad getters and setters in order to access the attribute of the class.

### Q43. Differentiate between .py and .pyc files?
Both .py and .pyc files holds the byte code. “.pyc” is a compiled version of Python file. This file is automatically generated by Python to improve performance. The .pyc file is having byte code which is platform independent and can be executed on any operating system that supports .pyc format.
Note: there is no difference in speed when program is read from .pyc or .py file; the only difference is the load time.

### Q44. How to retrieve data from a table in MySQL database through Python code? Explain.
 import MySQLdb module as : import MySQLdb
 establish a connection to the database.
db = MySQLdb.connect(“host”=”local host”, “database-user”=”user-name”, “password”=”password”, “database-name”=”database”)
 initialize the cursor variable upon the established connection: c1 = db.cursor()
 retrieve the information by defining a required query string. s = “Select * from dept”
 fetch the data using fetch() methods and print it. data = c1.fetch(s)
 close the database connection. db.close()

### Q45. Explain about ODBC and Python ?
ODBC (“Open Database Connectivity) API standard allows the connections with any database that supports the interface, such as PostgreSQL database or Microsoft Access in a transparent manner . There are 3 ODBC modules for Python:

PythonWin ODBC module – limited development
mxODBC – commercial product
pyodbc – it is an open source Python package.

### Q46. How would you define a protected member in a Python class?
All the members of a class in Python are public by default. You don’t need to define an access specifier for members of class. By adding ‘_’ as a prefix to the member of a class, by convetion you are telling others please don’t this object, if you are not a subclass the respective class.
Eg: class Person:
empid = None
_salary = None #salary is a protected member & it can accessible by the subclasses of Person
….

### Q47. How do you remove duplicates from a list?
a. sort the list
b. scan the list from the end.
c. while scanning from right-to-left, delete all the duplicate elements from the list

### Q48. Differentiate between append() and extend() methods. ?
Both append() and extend() methods are the methods of list. These methods a re used to add the elements at the end of the list.
append(element) – adds the given element at the end of the list which has called this method.
extend(another-list) – adds the elements of another-list at the end of the list which is called the extend method.

### Q49. Name few Python Web Frameworks for developing web applications?
There are various web frameworks provided by Python. They are
web2py – it is the simplest of all the web frameworks used for developing web applications.
cherryPy – it is a Python-based Object oriented Web framework.
Flask – it is a Python-based micro-framework for designing and developing web applications.

### Q50. How do you check the file existence and their types in Python?
os.path.exists() – use this method to check for the existence of a file. It returns True if the file exists, false otherwise. Eg: import os; os.path.exists(‘/etc/hosts’)
os.path.isfile() – this method is used to check whether the give path references a file or not. It returns True if the path references to a file, else it returns false. Eg: import os; os.path.isfile(‘/etc/hosts’)
os.path.isdir() – this method is used to check whether the give path references a directory or not. It returns True if the path references to a directory, else it returns false. Eg: import os; os.path.isfile(‘/etc/hosts’)
os.path.getsize() – returns the size of the given file
os.path.getmtime() – returns the timestamp of the given path.

### Q51. Name few methods that are used to implement Functionally Oriented Programming in Python?
Python supports methods (called iterators in Python3), such as filter(), map(), and reduce(), that are very useful when you need to iterate over the items in a list, create a dictionary, or extract a subset of a list.
filter() – enables you to extract a subset of values based on conditional logic.
map() – it is a built-in function that applies the function to each item in an iterable.
reduce() – repeatedly performs a pair-wise reduction on a sequence until a single value is computed.

