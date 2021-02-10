import os
from dotenv import load_dotenv, find_dotenv, set_key

filedir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(filedir)


def testsetup(path_to_s7kfile: str):
    with open(os.path.join(root_dir, ".env"), "w") as envfile:
        envfile.write(f"s7k_filepath={path_to_s7kfile}")
    

# A s7k file is needed to test the pyread library. If no such file exists the tests will fail.
# If you have some s7k data, then the following condition creates an .env file 
def check_for_s7kfile(max_recursions=3):
    if max_recursions==0:
        raise RecursionError("You failed to specify a correct path in a reasonable time. Try again.")
    path2env = find_dotenv()
    if path2env == "":
        print("No .env file exists. Creating one...")
        path_to_s7kfile = input("Absolute path to a s7k file: ")
        testsetup(path_to_s7kfile)

    path2env = find_dotenv()
    load_dotenv(path2env)
    path_to_s7kfile= os.environ.get("s7k_filepath", None)
    if path_to_s7kfile is None:
        print("No path to an s7k file exists in the .env file, appending to existing .env file")
        path_to_s7kfile = input("Absolute path to a s7k file: ")
        set_key(path2env, "s7k_filepath", path_to_s7kfile)
    elif path_to_s7kfile.endswith(".s7k") and os.path.exists(path_to_s7kfile):
        return True
    check_for_s7kfile(max_recursions-1)
    

check_for_s7kfile()