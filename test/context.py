import os
from dotenv import load_dotenv, find_dotenv, set_key

filedir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(filedir)


def testsetup(bf_filepath: str, iq_filepath: str):
    with open(os.path.join(root_dir, ".env"), "w") as envfile:
        envfile.write(f"bf_filepath={bf_filepath}\n")
        envfile.write(f"iq_filepath={iq_filepath}")


# A s7k file is needed to test the pyread library. If no such file exists the tests will fail.
# If you have some s7k data, then the following condition creates an .env file
def check_for_s7kfile(max_recursions=3):
    if max_recursions == 0:
        raise RecursionError(
            "You failed to specify correct paths in a reasonable time. Try again."
        )
    path2env = find_dotenv()
    if path2env == "":
        print("No .env file exists. Creating one...")
        bf_filepath = input("Absolute path to a s7k file with beamformed data: ")
        iq_filepath = input("Absolute path to a s7k file with IQ data: ")
        testsetup(bf_filepath=bf_filepath, iq_filepath=iq_filepath)

    path2env = find_dotenv()
    load_dotenv(path2env)
    bf_filepath = os.environ.get("bf_filepath", None)
    iq_filepath = os.environ.get("iq_filepath", None)
    if any([bf_filepath is None, iq_filepath is None]):
        print("Missing path to either IQ or beamformed data in the .env file...")
        bf_filepath = input("Absolute path to a s7k file with beamformed data: ")
        iq_filepath = input("Absolute path to a s7k file with IQ data: ")
        set_key(path2env, "bf_filepath", bf_filepath)
        set_key(path2env, "iq_filepath", iq_filepath)
    elif (bf_filepath.endswith(".s7k") and iq_filepath.endswith(".s7k")) and all(
        map(os.path.exists, [bf_filepath, iq_filepath])
    ):
        return True
    check_for_s7kfile(max_recursions - 1)


check_for_s7kfile()

bf_filepath = os.environ.get("bf_filepath", None)
iq_filepath = os.environ.get("iq_filepath", None)
if any([bf_filepath is None, iq_filepath is None]):
    raise ValueError(
        "Unexpected ValueError! s7k_filepath is for some reason not defined!"
    )
