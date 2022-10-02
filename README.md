# calgen by Sergio Eraso
## A lightweight, flexible, on-call shift scheduler

This project was made for senior resident advisors (SRAs) at Emory University. 
SRAs are responsible for creating monthly on-call schedules for their RAs. These schedules
are subject to various constraints such as respecting the availability of RAs, maintaining
a fair distribution of shifts, and not overwhelming a particular RA with back-to-back on-call shifts.
Considering that many staffs have over a dozen RAs, the SRAs task of creating a monthly calendar that satisfies these
constraints can be a lengthy and tedious process. Additonally, staffs often have different policies about which shifts carry
more weight. For instance, staffs usually regard weekend shifts as a larger responsibility, but the exact days that are considered "weekends"
vary across staffs. For these reasons, I created a lightweight, customizable, on-call shift scheduler that can generate these on-call
calendars in seconds, satisfying all of the constraints mentioned (given reasonable RA availability information).
