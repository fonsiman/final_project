from git import Repo

PATH_OF_GIT_REPO = r'/home/alfonso/ironhack/final_project/.git'  # make sure .git folder is properly configured
COMMIT_MESSAGE = 'Guardado automático por gestos'

def git_push():
    try:

        repo = Repo(PATH_OF_GIT_REPO)  # if repo is CWD just do '.'
        repo.index.add(["main.py"])
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote('origin')
        origin.push()

        print('Added, commited & pushed')

    except:
        print('Some error occured while pushing the code')

git_push()
