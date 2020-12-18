from num2words import num2words
from class_string_formatting import *

def format_normal(name,last_name,age,profession):
    print("Hello, %s %s. You are %s and you are a %s." %(name,last_name,age,profession))
    print('\n')

def format_format(name,last_name,age,profession):
    print("Hello, {} {}. You are {} and you are a {}." .format(name,last_name,age,profession))
def format_format2(name,last_name,age,profession):
    print("test {name}".format(name=age))
    print("Hello, {name} {last_name}. You are {age} and you are a {profession}." .format(name=name,last_name=last_name,age=age,profession=profession))
def format_format_dict(dict_name_age):
    print("You are {} and are {} years old.".format(dict_name_age['name'],dict_name_age['age']))
def format_format_dict2(dict_name_age):
    print("You are {name} and are {age} years old.\n".format(**dict_name_age))

def turn_to_word_number(number):
    return num2words(number)

def format_f_format(name,last_name,age,profession):
    print(f"Hello, {name} {last_name}. You are {age} and you are a {profession}.")
    print(f"Hello, {name.upper()} {last_name.lower()}. You are {turn_to_word_number(age)} and you are a {profession.capitalize()}.")

def main():
    name = "Nattapat"
    last_name = "Jutha"
    age = 24
    profession = "student"
    dict_name_age = {'name': name, 'age': age}

    format_normal(name,last_name,age,profession)

    format_format(name,last_name,age,profession)
    format_format2(name,last_name,age,profession)
    format_format_dict(dict_name_age)
    format_format_dict2(dict_name_age)

    format_f_format(name,last_name,age,profession)

    student_1 = Student("Chris", "Twen",33)
    print(student_1)
    print(f"{student_1}")
    print(f"{student_1!r}")

if __name__ == "__main__":
    main()
