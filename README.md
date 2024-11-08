# GitHub Actions  ===> 
# This is a powerful automation tool provided by Github that enables developers to automate tasks related to their software development workflows . It allows for the creation of custom workflows , which can be triggered by various events , such as pushes to a repository , pull requests or scheduled events . These workflows can be used for various purposes , including Continuous Integration(CI) , Continuous Deployment(CD), testing and more . 


# GitHub ===> Code Repository ===> Commit Code ===> Collaborative . 

# Git ===> Distributed Version Control System[Developers] ===> Track source Code files . 

# GitHub Action ===> CI/CD Tool 
# GitHub Actions  ===> 
# This is a powerful automation tool provided by Github that enables developers to automate tasks related to their software development workflows . It allows for the creation of custom workflows , which can be triggered by various events , such as pushes to a repository , pull requests or scheduled events . These workflows can be used for various purposes , including Continuous Integration(CI) , Continuous Deployment(CD), testing and more . 


# GitHub ===> Code Repository ===> Commit Code ===> Collaborative . 

# Git ===> Distributed Version Control System[Developers] ===> Track source Code files . 

# GitHub Action ===> CI/CD Tool 


# Introduction to CI(Continuous Integration) And CD(Continuous Deployment) ===> 
#  CI and CD are 2 keys practices in modern software development and GitHub Actions can facilitate both : 
# 
# (a). (CI) ===> CI  is a prectice where developers frequently merge their code changes into a shared repository. Each merge triggers an automated build and testing process to ensure the changes do not break the existing functionality. With GitHub Actions , developers can set up workflows to automatically build and test their code everytime they push changes to the repository or create a pull request . 
# 
# 
# (b). (CD) ===> CD extends the concept of CI  by automating the deployment of code to production environments after it passes all required tests . GitHub Actions can be configured to automatically deploy applications to various environments , such as staging or production , once the code passes CI checks. This prectice reduce the time between development and deployment , allowing for faster delivery of features and updates . 


# =====>>>>> Benefits of a Devepoer's Workflow :::
# (1). Improved Collaboration : Clear workflows  and processes facilitate better collaboration among team members , making it easier to understand who is responsible for what . 
# (2). Higher Code Quality : Code reviews , automated tests , and CI/CD practices help maintain a high standard of code quality . 
# (3). Reduced Errors : Automated testing and deployment reduce the likelihood of human error , ensuring more reliable releases . 
# (4). Faster Delivery : Streamlined workflows enable faster development and release cycles, allowing teams to deliver new features and fixes more quickly . 
# (5). Continuous Feedback Loop : Regular monitoring and feedback help teams quickly idnetify and address issues , continuously improving the product . 


Steps 
(1) create a folder ===> create a README.md file 
open github ===> create repo ==> 
git add README.md 
git init
git add . 
git push -u origin Main 
git remote add origin https://github.com/username.git 
git push -u origin main 


create a 'src' folder ===> (1) __init__.py (2) math_operations.py 
create a 'tests' folder ===> (1) __init__.py (2) test_math_operations

in vs code ==> extensions ==> github Actions Install via github 

go to github ===> Actions ===> create and test python application configure ===> 
.github/workflows/python-app.yml 

in vs code ===> .github/workflows/unittest.yml ===> 

copy unittest.yml script and paste it into python-app.yml file and click to commit changes 
click on code then click on actions ==> then we will see our all running  workflows . 