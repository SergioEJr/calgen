![<h1>A lightweight, flexible, on-call shift scheduler</h1>](https://github.com/SergioEJr/calgen/blob/master/calgen_title.PNG?raw=true)
## A lightweight, flexible, on-call shift scheduler
This project was made for senior resident advisors (SRAs) at Emory University. 
SRAs are responsible for creating monthly on-call schedules for their RAs. These schedules
are subject to various hard and soft constraints such as respecting the availability of RAs, maintaining
a fair distribution of shifts, and not overwhelming a particular RA with back-to-back on-call shifts.
Considering that many staffs have over a dozen RAs, the SRAs task of creating a monthly calendar that satisfies these
constraints can be a lengthy and tedious process. Additonally, staffs often have different policies about which shifts carry
more weight. For these reasons, I created a lightweight, customizable, on-call shift scheduler that can generate these on-call
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
Here are some of the attributes we expect from a good calendar-generating algorithm:
- Shifts should be distributed "fairly" both locally (at the calendar level) and globally (at the academic-year-long level)
- Adjacent days with the same RA on-call are undesireable
- Availability conflicts should be avoided as much as possible

A quick note: It seems that we would ideally want the last point to be a hard constraint -- any calendar that contains availability conflicts is automatically invalid. This would be true if we were scheduling doctors at a hospital, rather than RAs. However, adding a hard constraint can bring many issues. For instance, how do we handle RAs with very low availability (i.e. strong scheduling preferences)? With a hard constraint, RAs could "game the system" by purposely having extremely strong scheduling preferences, thus always obtaining their desired on-call days. This would not be fair to other RAs that fill out availability forms honestly. Thus, it is clear that we must strike a balance between respecting RA availabilities and maintaining a fair workload among all the RAs.

We must find a way to quantify each one of these constraints taking into account the fact that weekdays and weekends have different values. Thus, we adopt a point system to quantify the workload of a shift. By default, weekdays have a value of 1 and weekends have a value of 2 (can be changed by the user). We represent a calendar with a 1D array of RA objects. Each RA object will have a 1D binary array associated with it corresponding to their availability. Each RA object also has attributes `local_points` and `global_points`. The former quantifies the workload of an RA for the current calendar while the latter quantifies the historical work the RA has completed. Therefore, to quantify the "fairness" of a particular calendar, we can simply calculate the standard deviation of the set of ``local_points`` of all RAs. For example, if a particular calendar has a point distribution {"Alice" : 5, "Bob" : 3, "Charlie" : 10}, it would be considered to be less fair than the distribution {"Alice" : 6, "Bob" : 6, "Charlie" : 6}. The global "fairness" can be quantified the same way, but now by using the historical data ``global_points``.

For this implementation, all constraints $s_1,\dots, s_n$ are quantified in a similar way. Thus, each constraint has an associated cost ${c_1, \dots, c_n}$ where the $c_i = c_i(s_i)$. Therefore, the quality of a calendar can be quantified by $\sum w_i c_i$ where $w$ is some weight vector.

Essentially, what this program does is it samples the space of possible calendars and returns the calendar with the lowest total cost.

## Documentation
Documentation can be accessed from inside the app by typing 'help'.
