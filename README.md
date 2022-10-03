![<h1>A lightweight, flexible, on-call shift scheduler</h1>](https://github.com/SergioEJr/calgen/blob/master/calgen_title.PNG?raw=true)

This project was made for senior resident advisors (SRAs) at Emory University. 
SRAs are responsible for creating monthly on-call schedules for their RAs. These schedules
are subject to various constraints such as respecting the availability of RAs, maintaining
a fair distribution of shifts, and not overwhelming a particular RA with back-to-back on-call shifts.
Considering that many staffs have over a dozen RAs, the SRAs task of creating a monthly calendar that satisfies these
constraints can be a lengthy and tedious process. Additonally, staffs often have different policies about which shifts carry
more weight. For instance, staffs usually regard weekend shifts as a larger responsibility, but the exact days that are considered "weekends"
vary across staffs. For these reasons, I created a lightweight, customizable, on-call shift scheduler that can generate these on-call
calendars in seconds, satisfying all of the constraints mentioned (given reasonable RA availability information).

![<h2>Features</h2>](https://github.com/SergioEJr/calgen/blob/master/calgen_example.PNG?raw=true)

## Features
- Easy-to-use command-line interface with built-in documentation
- Staff Management
  - add/remove RAs from your staff
  - view total points for each RA
  - edit which days count as weekends
  - edit how many points weekends are worth
- Calendar Creation
  - ML generated calendar given an availability CSV
  - availability conflict warnings
  - breakdown of shift distribution
  - can edit the generated calendar to your liking
  - export calendar to CSV (excel-readable file)

## How it works
TODO

## Documentation
Documentation can be accessed from inside the app by typing 'help'.
