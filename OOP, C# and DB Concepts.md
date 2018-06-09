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
| Interface                                                                                                                                                | Abstract Class                                                                                                                                                                  |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A class may implement one or more interfaces.                                                                                                            | A class may inherit only one Abstract Class or any other class                                                                                                                  |
| An interface cannot have access modifiers for any types declared in it. By Default all are public.                                                       | An abstract class can contain  access modifiers.                                                                                                                                |
| If various implementations only share method signatures then it is better to use Interfaces.                                                             | If various implementations are of the same kind and use common behavior or status then abstract class is better to use.                                                         |
| Requires more time to find the actual method in the corresponding classes.                                                                               | Fast                                                                                                                                                                            |
| All methods declared in interface must be implemented in derived class.                                                                                  | Only abstract methods need to be implemented in derived classes.                                                                                                                |
| If we add a new method to an Interface then we have to track down all the implementations of the interface and define implementation for the new method. | If we add a new method to an abstract class then we have the option of providing default implementation and therefore all the existing code will work without any modification. |
| No fields can be defined in interfaces                                                                                                                   | We can define fields and constants in abstract class.                                                                                                                           |