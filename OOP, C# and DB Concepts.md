# OOP, C# and DB Concepts:
### Q1. What is unsafe code?
ANS: To maintain type safety and security, by default C# does not support pointer arithmetic. However, by using the unsafe keyword, it is possible to define an unsafe context in which pointers can be used.

### Q2. What are the properties of unsafe code?	
ANS: 
- Methods, types, and code blocks can be defined as unsafe.
- In some cases, unsafe code may increase an application's performance by removing array bounds checks.
- Unsafe code is required when calling native functions that require pointers.
- Using unsafe code introduces security and stability risks.
- In order for C# to compile unsafe code, the application must be compiled with /unsafe.

### Q3. What is dangling pointer?
ANS: A dangling pointer arises when you use the address of an object after its lifetime is over.

### Q4. What is this pointer?
ANS: this pointer is a pointer which points to the current object of a class. this is actually a keyword which is used as a pointer which differentiate the current object with global object.

### Q5. What is anonymous function?
ANS: An anonymous function has no name. By using the delegate keyword, you can add multiple statements inside your anonymous function.

### Q6. What are delegates?
ANS: Delegates in the C# language are objects that reference a certain behavior.

### Q7. What are sealed classes?
ANS: The classes declared with sealed keyword cannot be inherited.

### Q8. What are abstract classes?
ANS: The abstract modifier is used in a class declaration to indicate that a class is intended only to be a base class of other classes.	

### Q9. What are the features of abstract methods?
ANS: Abstract methods have no implementations. The implementation logic is provided rather by classes that derive from them.
Features: 
- An abstract class cannot be instantiated.
- An abstract class may contain abstract methods and accessors.
- It is not possible to modify an abstract class with the sealed modifier, which means that the class cannot be inherited.
- A non-abstract class derived from an abstract class must include actual implementations of all inherited abstract methods and accessors.

### Q10. What is extern modifier?
ANS: The extern modifier is used in a method declaration to indicate that the method is implemented externally.

### Q11. What is static class and static method?
ANS: 

Static Classes:
- Static classes and methods have no references.
- Static classes never instantiated. 
- The static keyword on a class enforces that a type not be created with a constructor. 
- In the static class, we access members directly on the type.
- A static class cannot have non-static members.

Static Methods:
- Static methods have no instances. They are called with the type name, not an instance identifier.
- Static methods are called without an instance reference.
- Static methods cannot access non-static class level members and do not have a 'this' pointer.

### Q12. What are interfaces?
ANS: An interface describes a contract. Interfaces contain methods without implementation. They can be used across many different classes. It becomes possible to call the methods through the interface reference.

### Q13. What is the virtual keyword in .net?
ANS: The virtual keyword is used to modify a method, property, indexer, or event declaration and allow for it to be overridden in a derived class. The implementation of a virtual member can be changed by an overriding member in a derived class.

### Q14. Interface VS Abstract?
ANS: 

Interface
- A class may implement one or more interfaces.
- An interface cannot have access modifiers for any types declared in it. By Default all are public.
- If various implementations only share method signatures then it is better to use Interfaces.
- Requires more time to find the actual method in the corresponding classes.
- All methods declared in interface must be implemented in derived class.
- If we add a new method to an Interface then we have to track down all the implementations of the interface and define implementation for the new method.
- No fields can be defined in interfaces

Abstract class
- A class may inherit only one Abstract Class or any other class
- An abstract class can contain access modifiers.
- If various implementations are of the same kind and use common behavior or status then abstract class is better to use.
- Fast
- Only abstract methods need to be implemented in derived classes.
- If we add a new method to an abstract class then we have the option of providing default implementation and therefore all the existing code will work without any modification.
- We can define fields and constants in abstract class.

### Q15. What is the difference between class and structure?
|    Class                                    |    Structure                          |
|---------------------------------------------|---------------------------------------|
|    Reference   types                        |    Value   type                       |
|    Uses   Heap memory                       |    Uses   Stack Memory                |
|    Passed   by reference to the method      |    Passed   by value to the method    |
|    Field   Initialize ability               |    No   field initialize ability      |
|    Explicit   parameter less constructor    |    No                                 |
|    Inheritance                              |    No                                 |


### Q16. What are association, aggregation and composition?
ANS:
- Association is a relationship where all object have their own lifecycle and there is no owner. Let’s take an example of Teacher and Student. Multiple students can associate with single teacher and single student can associate with multiple teachers but there is no ownership between the objects and both have their own lifecycle. Both can create and delete independently.
- Aggregation is a specialize form of Association where all object have their own lifecycle but there is ownership and child object cannot belongs to another parent object. Let’s take an example of Department and teacher. A single teacher cannot belong to multiple departments, but if we delete the department teacher object will not destroy. We can think about “has-a” relationship.
- Composition is again specialize form of Aggregation and we can call this as a “death” relationship. It is a strong type of Aggregation. Child object does not have their lifecycle and if parent object deletes all child objects will also be deleted. Let’s take again an example of relationship between House and rooms. House can contain multiple rooms there is no independent life of room and any room cannot belongs to two different house if we delete the house room will automatically delete. Let’s take another example relationship between Questions and options. Single questions can have multiple options and option can not belong to multiple questions. If we delete questions options will automatically delete.

### Q17. What is class?
ANS: A Class is simply a chunk of code that does a particular job. The idea is that you can reuse this code (Class) whenever you need it, or when you need it in other projects. It saves you from having to write the same thing over and over again. Class describes the type of object.

### Q18. What is Object?
ANS: Objects are useful instances of the classes. So, the act of creating an object is called instantiation.
Three properties characterize objects:
1. Identity: the property of an object that distinguishes it from other objects
2. State: describes the data stored in the object
3. Behavior: describes the methods in the object's interface by which the object can be used

### Q19. What is object class?
ANS:  Provides low level services to the derived classes. This is the ultimate base class of all classes in the .NET Framework; it is the root of the type hierarchy. Derived classes can and do override some of these methods, including: 
- Equals- Supports comparisons between objects.
- Finalize- Performs cleanup operations before an object is automatically reclaimed.
- GetHashCode- Generates a number corresponding to the value of the object to support the use of a hash table.
- ToString- Manufactures a human-readable text string that describes an instance of the class.

### Q20. What is the difference between ExecuteReader(), ExecuteScalar(), ExecuteNonQuery() and ExecuteQuery()?
ANS: 
- ExecuteScalar is going to be the type of query which will be returning a single value. An example would be selecting a count of the number of active users.
- ExecuteReader gives you a data reader back which will allow you to read all of the columns of the results a row at a time. An example would be getting all of the information for each user in the system so you could display that information.
- ExecuteNonQuery is any SQL which isn't returning values really, but is actually performing some form of work like inserting deleting or modifying something. An example would be updating a user's personal information in the database.
- ExecuteQuery: Executes SQL queries directly on the database and returns objects.

### Q21. What is the difference between DataSet, DataTable, DataView and DataAdapter?
ANS:
- DataReader – The data reader is a forward-only, read only stream of data from the database. This makes the data reader a very efficient means for retrieving data, as only one record is brought into memory at a time. The disadvantage: A connection object can only contain one data reader at a time, so we must explicitly close the data reader when we are done with it.
- DataAdapter – Represents a set of SQL commands and a database connection that are used to fill the Data Set and update the data source. It serves as a bridge between a Data Set and a data source for retrieving and saving data. 
- DataSet – The Data Set is the centerpiece of a disconnected, data-driven application; it is an in-memory representation of a complete set of data, including tables, relationships, and constraints. The Data Set does not maintain a connection to a data source, enabling true disconnected data management. The data in a Data Set can be accessed, manipulated, updated, or deleted, and then reconciled with the original data source. Since the Data Set is disconnected from the data source, there is less contention for valuable resources, such as database connections, and less record locking.
- DataView – A major function of the Data View is to allow for data binding on both Windows Forms and Web Forms. Usually, a Data View data component is added to the project to display a sub set of the records in a Data Set or Data Table. It also acts as filtered and sorted view of data to the Data Table.
- DataTable – Represents one table of in memory data. A Data Table stores data in a similar form to a database table.

### Q22. What is web service?
ANS: Web services are XML-based information exchange systems that use the Internet for direct application-to-application interaction. These systems can include programs, objects, messages, or documents.

### Q23. What is SOAP?
ANS: SOAP is a simple XML-based protocol that allows applications to exchange information over HTTP.

### Q24. What is WSDL?
ANS: WSDL is an XML-based language for describing Web services and how to access them.

### Q25. What is UDDI?
ANS: UDDI is a directory service where companies can register and search for Web services.

### Q26. What are the benefits of the web services?
- Easier to communicate between applications 
- Easier to reuse existing services 
- Easier to distribute information to more consumers 
- Rapid development 

### Q27. What is the architecture of the web service?
ANS:
- Service provider: 
This is the provider of the web service. The service provider implements the service and makes it available on the Internet.
- Service requestor 
This is any consumer of the web service. The requestor utilizes an existing web service by opening a network connection and sending an XML request.
- Service registry 
This is a logically centralized directory of services. The registry provides a central place where developers can publish new services or find existing ones. It therefore serves as a centralized clearinghouse for companies and their services.

### Q28. What is Normalization?
ANS: Normalization is the process of efficiently organizing data in a database. There are two goals of the normalization process: 
1.  Eliminating redundant data
2.  Ensuring data dependency
- 1st Normal form: Remove repeating groups.
- 2nd Normal form: Remove partial dependency.
- 3rd Normal form: Remove non key attribute dependency.

### Q29. What is the weak reference in C#?
ANS: A weak reference allows the garbage collector to collect an object while still allowing an application to access the object.

### Q30. Can we use private constructor? 
ANS:  Yes we can use private constructor, it is similar to public constructor. The private constructor simply makes it impossible for external code to instantiate the class directly.

### Q31. What are singleton classes?
ANS: Singleton Pattern ensures that a class has only one instance and provides a global point of access to it.

### Q32. Can we use more than one finally block for nested try catch?
ANS: Yes

### Q33. Can we store data in an array of more than one data type?
ANS: No

### Q34. How can we store different types of data in an array?
ANS: Using object array and array list we can store multiple data types data in an array.

### Q35. What is the difference between HAVING and WHERE?
ANS: A HAVING clause is like a WHERE clause, but applies only to groups as a whole, whereas the WHERE clause applies to individual rows. HAVING can only be used with SELECT statement. We cannot use HAVING clause without GROUPBY clause. WHERE keyword could not be used with aggregate functions but HAVING can be used. HAVING can be used without WHERE clause.

### Q36. What is the result of select * from table where 1=1?
ANS: It returns the all data from table. Because 1=1 condition always be true. It is equivalent to select * from table.

### Q37. What is boxing and un-boxing in ASP.NET/C#?
ANS: When a variable of a value type is converted to object, it is said to be boxed. When a variable of type object is converted to a value type, it is said to be unboxed. Boxing and un-boxing enable value types to be treated as objects.

### Q38. What is Reflection?
ANS: The ability to discover the methods and fields in a class as well as invoke methods in a class at runtime, typically called reflection. Reflection in C# achieved at assembly level.

### Q39. What is Page Life Cycle?
ANS: The page event life cycle consists of the following page events, which occur in the following order:
- Page request: The page request occurs before the page life cycle begins. When the page is requested by a user, ASP.NET determines whether the page needs to be parsed and compiled or whether a cached version of the page can be sent in response without running the page.
- Start: In the start step, page properties such as Request and Response are set.
- Page initialization: This page event initializes the page by creating and initializing the web server controls on the page.
- Load: This event runs every time the page is requested.
- Validation: During validation, the Validate method of all validator controls is called, which sets the ‘IsValid’ property of individual validator controls and of the page.
- Post back event handling: If the request is a postback, any event handlers are called.
- Rendering: During the rendering phase, the page calls the Render method for each control, providing a text writer that writes its output to the OutputStream of the page’s Response property.
- Unload: Unload is called after the page has been fully rendered, sent to the client, and is ready to be discarded.

### Q40. What are Partial classes in .NET?
ANS: With partial, you can physically separate a class into multiple files. This is often done by code generators.

### Q41. What is the difference between Server.Transfer and Response.Redirect?  
ANS: In Server.Transfer page processing transfers from one page to the other page without making a round-trip back to the client’s browser.  This provides a faster response with a little less overhead on the server.  The clients url history list or current url Server does not update in case of Server.Transfer.
Response.Redirect is used to redirect the user’s browser to another page or site.  It performs trip back to the client where the client’s browser is redirected to the new page.  The user’s browser history list is updated to reflect the new address.
From which base class all Web Forms are inherited?
All web forms are inherited from Page class. 

### Q42. What are the different validators in ASP.NET?
ANS:
- Required field Validators
- Range  Validators
- Compare Validators
- Custom Validators
- Regular expression Validators
- Summary Validators

### Q43. What is ViewState?
ANS: ViewState is used to retain the state of server-side objects between page post backs.

### Q44. Where the viewstate is stored after the page postback?
ANS: ViewState is stored in a hidden field on the page at client side.  ViewState is transported to the client and back to the server, and is not stored on the server or any other external source.

### Q45. What are the different Session state management options available in ASP.NET?
ANS: There are two session state management options available in asp.net; In-Process and Out-of-Process.
In-Process stores the session in memory on the web server. 
Out-of-Process Session state management stores data in an external server.  The external server may be either a SQL Server or a State Server.  All objects stored in session are required to be serializable for Out-of-Process state management.

### Q46. What is caching?
ANS: Caching is a technique used to increase performance by keeping frequently accessed data or files in memory. The request for a cached file/data will be accessed from cache instead of actual location of that file.

### Q47. What are the different types of caching?
ANS: ASP.NET has 3 kinds of caching:
1. Output Caching,
2. Fragment Caching,
3. Data Caching.

### Q48. Which type if caching will be used if we want to cache the portion of a page instead of whole page?
ANS: Fragment Caching: It caches the portion of the page generated by the request. For that, we can create user controls with the below code:
	<%@ OutputCache Duration="120" VaryByParam="CategoryID;SelectedID"%>

### Q49. Can we have a web application running without web.Config file?
ANS: Yes

### Q50. Is it possible to create web application with both webforms and mvc?
ANS: Yes. We have to include below MVC assembly references in the web forms application to create hybrid application.

### Q51. What is the difference between web config and machine config?
ANS: Web config file is specific to a web application where as machine config is specific to a machine or server. There can be multiple web config files into an application where as we can have only one machine config file on a server.

### Q52. What is the difference between MVC and 3-tier model?
ANS: At first glance, the three tiers may seem similar to the MVC (Model View Controller) concept; however, topologically they are different. A fundamental rule in three-tier architecture is the client tier never communicates directly with the data tier; in a three-tier model all communication must pass through the middleware tier. Conceptually the three-tier architecture is linear. However, the MVC architecture is triangular: the View sends updates to the Controller, the Controller updates the Model, and the View gets updated directly from the Model.

### Q53. Which namespaces are necessary to create a localized application?
ANS:
- System.Globalization 
- System.Resources

### Q54. What are the different types of cookies in ASP.NET?
ANS: Session Cookie Resides on the client machine for a single session until the user does not log out.
Persistent Cookie – Resides on a user’s machine for a period specified for its expiry, such as 10 days, one month, and never.

### Q55. What is the file extension of web service?
ANS: Web services have file extension .asmx.

### Q56. What are the components of ADO.NET?
ANS: The components of ADO.Net are Dataset, Data Reader, Data Adaptor, Command, connection.

### Q57. How to retrieve the 2nd highest record from the table? Write query.
ANS: Select max(Colomn_Value) from Table where Colomn_Value < (select max(Colomn_Value) from Table)
