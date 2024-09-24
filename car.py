i=1
started = False
while i==1:
    order = input(">")
    if order == "help":
        print("start - to start the car\nstop - to stop the car\nquit - to exit")
    elif order == "start":

        if not   started:
            started = True
            print("Car started...Ready to go!")
        else:
            print("already started")

    elif order == "stop":

        if  started:
            started = False
            print("Car stopped.")
        else:
            print("already stopped")

    elif order == "quit":
        break
    else:
        print("I don't understand")
