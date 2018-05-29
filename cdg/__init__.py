import cdg

def version():
    import subprocess
    command = "git --git-dir " + cdg.__path__[0][:-4] + "/.git rev-parse HEAD"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    cdg_git_commit_hash, err = process.communicate()
    return cdg_git_commit_hash, err

