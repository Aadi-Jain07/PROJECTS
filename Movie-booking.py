# Movie Ticket Booking System

available_seats = 20
ticket_price = 200

while True:
    print("\n===== Movie Ticket Booking System =====")
    print("1. Show Available Seats")
    print("2. Book Ticket")
    print("3. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        print("Available Seats:", available_seats)

    elif choice == 2:
        print("Available Seats:", available_seats)

        tickets = int(input("How many tickets do you want to book? "))

        if tickets <= available_seats:
            name = input("Enter your name: ")

            total_amount = tickets * ticket_price
            available_seats -= tickets

            print("\n----- Booking Confirmed -----")
            print("Customer Name:", name)
            print("Tickets Booked:", tickets)
            print("Total Amount: ₹", total_amount)
            print("Seats Left:", available_seats)

        else:
            print("Sorry! Not enough seats available.")

    elif choice == 3:
        print("Thank you for using the Movie Ticket Booking System!")
        break

    else:
        print("Invalid choice. Please try again.")
