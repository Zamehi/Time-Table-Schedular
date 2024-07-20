

import random
# --------------- TIME TABLE -----------------
class timetable:
    numberOfDays = 0
    startTime = "8:30"
    endTime = "5:30"

    def __init__(self, numberOfDays):
        self.numberOfDays= numberOfDays




# --------------- CLASSROOM ---------------
class classroom:
    noOfSeats = 0 #60 / 120
    def __init__(self, noOfSeats):
        self.noOfSeats = noOfSeats



# --------------- LECTURE -----------------
class lecture:
    def __init__(self, students, time, classroom, type):# type refers to lab or normal class , break
        self.students = students
        self.time= time
        self.classroom = classroom
        self.type = type



# ---------------- TEACHER -----------------
class teacher:
    def __init__(self, lecture, course, ):
        self.lecture = lecture
        self.course = course



# ----------------- SECTION -----------------
class section:
    def __init__(self, course ):
        self.course= course;



# ----------------- COURSE ------------------
# course is composed of lecture
class course :

    def __init__(self, lecture, sectio):
        self.lecture = lecture
        self.course = course



class Student:
    def __init__(self, course):
        self.course = course



class Teacher:
    def __init__(self, teacher_id, name, age, subject):
        self.teacher_id = teacher_id
        self.name = name
        self.age = age
        self.subject = subject

    def get_teacher_info(self):
        return f"Teacher ID: {self.teacher_id}, Name: {self.name}, Age: {self.age}, Subject: {self.subject}"

class Course:
    def __init__(self, course_id, name, teacher, students=[]):
        self.course_id = course_id
        self.name = name
        self.teacher = teacher
        self.students = students

    def add_student(self, student):
        self.students.append(student)

    def get_course_info(self):
        student_names = ", ".join([student.name for student in self.students])
        return f"Course ID: {self.course_id}, Name: {self.name}, Teacher: {self.teacher.name}, Students: {student_names}"
