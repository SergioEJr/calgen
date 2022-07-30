# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:45:06 2022

@author: sergi
"""

# pip
import numpy as np
from colorama import init, deinit, Fore, Style

# standard library
import calendar as cal
import pickle
import os
import sys
import time
import cmd
import re

# make errors in the terminal red
def eprint(string):
    print(f"{Fore.LIGHTRED_EX}{string}{Style.RESET_ALL}")

# organize RA instances into a Staff. the Staff has settings that apply
# to all RAs, such as which days count as weekends and how much a weekend
# is worth
class Staff:

    # initializes a staff with default settings
    def __init__(self, staff_name, names=None, points=None):
        self.staff_name = str(staff_name)
        # set up if no arguments provided
        if names is None:
            names = []
        if points is None:
            points = []
        if isinstance(names, str):
            names = [names]
        if isinstance(points, int):
            points = [points]
        # if no points provided, set all points to zero
        if len(points) < 1:
            points = np.zeros(len(names), dtype='float')
        if len(points) != len(names):
            raise ValueError("Names and points must be of same length")
        # numpy array of RA objects
        self.RAs = np.asarray(
            [RA(names[i], points[i]) for i in range(len(names))],
            dtype='object')
        # sorts RAs alphabetically
        self.RAs = Staff.sort_RAs(self.RAs)
        # the days which are considered weekends, Friday and Saturday by default
        self._weekends = np.asarray([4, 5])
        # the factor by which weekend points get multiplied by. 2 by default
        self._weekend_value = 2.0

    # prints a table sorted by name displaying name and points
    def __str__(self):
        return self.print_RAs(give=True)

    # returns array of name attributes of the RA objects
    def get_names(self):
        return np.asarray(
            [RA.name for RA in self.RAs])

    # returns array of point attributes of the RA objects
    def get_points(self):
        return np.asarray(
            [RA.points for RA in self.RAs])

    @property
    def weekends(self):
        return self._weekends

    @weekends.setter
    # checks if the ints given are in the valid range
    def weekends(self, new_weekends):
        '''
        Sets the days that count as weekend on-call shifts.
        
        Parameters
        ----------
        weekends : list of int in {0,1,2,3,4,5,6}
            0 corresponds to Monday and 6 corresponds to Sunday
        
        Returns
        -------
        None
        
        '''
        # remove duplicates
        new_weekends = np.unique(np.asarray(new_weekends))
        # if empty, set no weekends
        if new_weekends.size < 1:
            self._weekends = []
            print("No days will count as weekends")
            return
        # if not empty, set weekends and print confirmation
        result = ''
        for day_index in new_weekends:
            result += f"{cal.day_name[day_index]} "
        self._weekends = new_weekends
        print(f"Weekends changed to {result}")
        return

    @property
    def weekend_value(self):
        return self._weekend_value

    @weekend_value.setter
    def weekend_value(self, value):
        '''
        Sets the point factor of a weekend on-call shift.
        
        Parameters
        ----------
        value : float >= 1
            the factor by which weekend points will be multiplied
        Returns
        -------
        None
        
        '''
        self._weekend_value = float(value)
        print(f"Weekend value changed to {value}")

    # Sorts RAs by keyword sort
    # only for internal use
    @staticmethod
    def sort_RAs(RAs, sort='names'):
        '''
        Sorts the RA objects according to the keyword sort ('names' by default)
        
        Parameters
        ----------
        sort : str
            how the RA objects will be sorted, 'names' 'points' or 'pointsr'
            
        Returns
        -------
        sorted_RAs : array of RA objects sorted by the keyword sort
        '''
        if sort == 'names':
            names = np.asarray([RA.name for RA in RAs])
            return RAs[
                np.argsort(names)]
        elif sort == 'points':
            points = np.asarray([RA.points for RA in RAs])
            return RAs[
                np.argsort(points)[::-1]]
        elif sort == 'pointsr':
            points = np.asarray([RA.points for RA in RAs])
            return RAs[
                np.argsort(points)]
        else:
            raise ValueError("sort keyword must be 'names' 'points' or 'pointsr'")

    # prints a table sorted by either names, points, or reverse points
    def print_RAs(self, sort='names', give=False):
        names = self.get_names()
        if len(names) == 0:
            names = ['']
        lens_of_names = list(map(len, names))
        # find the longest name to know how to format the result
        max_len = max([15, max(lens_of_names)])
        # make a horizontal line
        hline = ''
        for i in range(2 * max_len):
            hline += '-'
        hline += '\n'
        res_advs = Staff.sort_RAs(self.RAs, sort)
        # make headers
        result = f"{'Name' : <{max_len}}{'Points' : >{max_len}}\n"
        result += hline
        # make table
        for RA in res_advs:
            result += f"{RA.name : <{max_len}}"
            result += f"{RA.points : >{max_len}.1f}\n"
        if give is True:
            return result
        else:
            print(result)

    # takes in a python array of names and points of the same length
    def add_RAs(self, names, points=None):
        '''
        Adds one or more RA objects to the current staff. If the current staff
        is not empty and no points are provided, all new RAs will be given the
        average number of points for an RA of the current staff.
        
        Parameters
        ----------
        names : str or list of str
            names of the RAs to be added to the staff
        points: int or list of int, optional
            points of the RAs to be added to the staff, must be of same size
            as names
        
        Returns
        -------
        None
        '''
        # handle single RA added
        if isinstance(names, str):
            names = [names]
        if isinstance(points, int):
            points = [points]
        current_names = set(self.get_names())
        for name in names:
            if name in current_names:
                raise ValueError(f"RA name '{name}' already taken.")
        for name in names:
            if not re.match(r'\w+', name):
                raise ValueError("Name must contain at least one word character")
        # if no points provided and staff is empty,
        # make all new RAs have zero points
        if points is None and self.RAs.size < 1:
            points = np.zeros(len(names))
        # if no points provided, but staff has some RAs, make all new RAs
        # have the average number of points in the current staff
        elif points is None:
            point_average = int(np.mean(self.get_points()))
            points = np.full(len(names), point_average)
        new_RAs = np.asarray([RA(name, point) for name, point in zip(names, points)])
        self.RAs = np.append(self.RAs, new_RAs)
        self.RAs = Staff.sort_RAs(self.RAs)

    # deletes one or more RA objects from the staff given their names
    def delete_RAs(self, names):
        '''
        Deletes one or more RAs from the current staff.
        
        Parameters
        ----------
        names : str or list of str
            names of the RAs to be removed from the current staff
        
        Returns
        -------
        None
        '''
        # handles single name case
        if isinstance(names, str):
            names = [names]
        curr_names = self.get_names()
        if curr_names.size == 0:
            raise ValueError("Cannot remove from an empty staff")
        # mask of RAs to remove
        mask = np.zeros(self.RAs.size, dtype='bool')
        for name in names:
            if name not in curr_names:
                raise ValueError(f"Invalid Name: '{name}' not found in staff")
            mask += curr_names == np.asarray(name)
        mask = np.invert(mask)
        # check if any names did not get removed
        self.RAs = self.RAs[mask]
        self.RAs = Staff.sort_RAs(self.RAs)

    # auxiliary function, only for internal use
    def prep_av(self, year, month, availability, off_days, assignments):
        '''Checks for common possible errors in the parameters of the 
        make_calendar function'''

        # if availability is a string, assume it is a path to a csv file
        # containing availabilities. Otherwise, availability is expected to be
        # a 2D binary array
        if isinstance(availability, str):
            try:
                availability = np.genfromtxt(availability, delimiter=',')
            except OSError:
                print("File not found. "
                      "You may need to include the file extention '.csv'. "
                      f"The current working directory is {os.getcwd()}")
        # make sure availability is a boolean array
        availability = np.asarray(availability, dtype='bool')
        # validate the sizes of availability
        avail_shape = np.shape(availability)
        num_res_advs = self.RAs.size
        num_days = cal.monthrange(year, month)[1]
        if avail_shape[0] != num_res_advs:
            error_message = f"Availability contains {avail_shape[0]} rows,"
            error_message += f" but there are {num_res_advs} RAs in the staff."
            raise ValueError(error_message)
        if avail_shape[1] != num_days:
            error_message = f"Availability contains {avail_shape[1]} columns,"
            error_message += f" but a calendar for {month}/{year} calls for"
            error_message += f" {num_days} columns."
            raise ValueError(error_message)
        # obtain all days to be assigned to check if there is intersection
        # with off days
        assigned_days = np.ravel(list(assignments.values()))
        if np.intersect1d(assigned_days, off_days).size > 0:
            raise ValueError("Off days and assignments cannot share days")
        # convert day numbers to indices in all arrays
        for key in assignments:
            assignments[key] = np.asarray(assignments[key]) - 1
        off_day_indices = np.asarray(off_days, dtype='int') - 1
        RAs = self.RAs
        names = self.get_names()
        # ensures that the availabilities given in assignments are accounted
        # for when calculating if an RA is low_availability
        for name in assignments:
            RA_index = np.argwhere(names == name)
            for day_index in assignments[name]:
                availability[RA_index, day_index] = True
        # set the availability for all RAs
        for i, row in enumerate(availability):
            RAs[i].availability = row

        return availability, off_day_indices, assignments

    def generate_calendar(self, year, month,
                          availability,
                          off_day_indices,
                          assignments,
                          n,
                          force_assignments=False,
                          momentum=0.2,
                          seed=None):
        costs = []
        if seed is not None:
            best_calendar = seed
            min_cost = best_calendar.cost
            costs.append(min_cost)
        else:
            best_calendar = None
            min_cost = 100000

        RAs = self.RAs
        names = self.get_names()
        weekend_value = self.weekend_value
        weekends = self.weekends
        curr_points = self.get_points()

        low_availability_threshold = np.mean(np.sum(availability, axis=1))
        - np.std(np.sum(availability, axis=1))

        for i in range(n):

            # set the initial state of all RA objects in the staff
            for i, RA in enumerate(RAs):
                RA.is_low_avail = (np.sum(availability[i])
                                   < low_availability_threshold)
                RA.is_overloaded = False
                RA.days_on_call = 0
                RA.weekends_on_call = 0
                RA.points = curr_points[i]

            # make calendar object
            cld = Calendar(year, month, RAs, availability)
            cld.weekends = self.weekends
            cld.weekend_value = self.weekend_value

            # set assignments in calendar
            assigned_indices = []
            for name in assignments:
                RA_index = np.argwhere(names == name)[0, 0]
                for day_index in assignments[name]:
                    # assign RA to the given days in assignments
                    cld.calendar[day_index] = RAs[RA_index]
                    # calculate points to be given
                    RAs[RA_index].days_on_call += 1
                    assigned_indices.append(day_index)
                    if cal.weekday(year, month, day_index + 1) in weekends:
                        RAs[RA_index].weekends_on_call += 1
                        RAs[RA_index].add_points(weekend_value)
                    else:
                        RAs[RA_index].add_points(1)

            # if not forcibly adding RAs to days when no one is available
            # remove days with zero availability from days to be used in the
            # calendar calculation

            not_off_day_mask = np.ones(cld.number_days, dtype='bool')
            not_assigned_mask = np.ones(cld.number_days, dtype='bool')
            for day_index in off_day_indices:
                not_off_day_mask[day_index] = False
            for day_index in assigned_indices:
                not_assigned_mask[day_index] = False
            if force_assignments is False:
                at_least_one_available_mask = np.any(availability, axis=0)
                mask = (at_least_one_available_mask 
                        * not_off_day_mask 
                        * not_assigned_mask)
            else:
                mask = not_off_day_mask * not_assigned_mask

            # remove off days and assigned days
            available_day_indices = np.arange(cld.number_days)[mask]

            # use the best_calendar so far as a seed
            if best_calendar is not None:
                num_placements = round(momentum * available_day_indices.size)
                # pick random days to inherit from best_calendar
                seed_indices = np.random.choice(available_day_indices
                                                , num_placements
                                                , replace=False)
                # set picked days to be the same as in the best_calendar
                for day_index in seed_indices:
                    day_week = cal.weekday(year, month, day_index + 1)
                    RA_in_best_cal = best_calendar.calendar[day_index]
                    cld.calendar[day_index] = RA_in_best_cal
                    RA_in_best_cal.days_on_call += 1
                    if day_week in weekends:
                        RA_in_best_cal.weekends_on_call += 1
                        RA_in_best_cal.add_points(weekend_value)
                    else:
                        RA_in_best_cal.add_points(1)
                # remove inherited days from available days
                mask = np.invert(np.isin(available_day_indices, seed_indices))
                available_day_indices = available_day_indices[mask]

            # fill in the calendar
            while available_day_indices.size > 0:

                # how many days on call is considered overloaded
                # decide which RAs are considered overloaded
                days_on_call = [RA.days_on_call for RA in RAs]
                mean_days_on_call = np.mean(days_on_call)
                std_days_on_call = np.std(days_on_call)
                overloaded_threshold = mean_days_on_call + 0 * std_days_on_call

                for RA in RAs:
                    if RA.days_on_call > overloaded_threshold:
                        RA.is_overloaded = True
                    else:
                        RA.is_overloaded = False

                # days with a small amount of availability are more likely to
                # get picked
                day_mask = np.zeros(cld.number_days, dtype='bool')
                for i in available_day_indices:
                    day_mask[i] = True

                day_factors = np.exp(
                    -1 * np.sum(availability[:, day_mask], axis=0))
                day_weights = day_factors / np.sum(day_factors)

                # choose a random available day
                # can include the argument p = day_weights to weight
                # days by availability
                # i'm not sure if this does anything
                day_index = np.random.choice(
                    available_day_indices,
                    p=day_weights)
                # see what kind of day it is
                day_week = cal.weekday(year, month, day_index + 1)

                # create subsets of available RAs
                # in order of priority:
                # not overloaded low availability RAs
                # not overloaded not low availability RAs
                # not overloaded RAs with low points
                availability_mask = availability[:, day_index]
                low_availability_mask = np.asarray(
                    [RA.is_low_avail for RA in RAs])
                not_overloaded_mask = np.asarray(
                    [not RA.is_overloaded for RA in RAs], dtype='bool')

                # not sure if i'll need these later
                # available_RAs = RAs[availability_mask]
                # low_av_available_RAs = RAs[
                #    low_availability_mask*
                #    availability_mask]

                not_over_low_av_available_RAs = RAs[
                    not_overloaded_mask *
                    low_availability_mask *
                    availability_mask]
                available_not_over_not_low_av_RAs = RAs[
                    np.invert(low_availability_mask) *
                    availability_mask *
                    not_overloaded_mask]
                not_overloaded_RAs = RAs[not_overloaded_mask]

                # not overloaded low availability available RAs
                if not_over_low_av_available_RAs.size != 0:
                    chosen_RA = np.random.choice(
                        not_over_low_av_available_RAs)
                # not overloaded or low availability available RAs
                elif available_not_over_not_low_av_RAs.size != 0:
                    chosen_RA = np.random.choice(
                        available_not_over_not_low_av_RAs)
                # not overloaded RAs regardless of availability
                else:
                    # weigh candidates with number of points
                    factors = np.exp(
                        [-1 * RA.points for RA in not_overloaded_RAs])
                    weights = factors / np.sum(factors)
                    chosen_RA = np.random.choice(
                        not_overloaded_RAs, p=weights)
                cld.calendar[day_index] = chosen_RA
                chosen_RA.days_on_call += 1
                if day_week in weekends:
                    points_to_add = weekend_value
                    chosen_RA.weekends_on_call += 1
                else:
                    points_to_add = 1
                chosen_RA.add_points(points_to_add)

                available_day_indices = np.delete(
                    available_day_indices, np.argwhere(
                        available_day_indices == day_index)[0])

            # calculate the cost of the calendar and compare to the
            # best cost so far
            cld.recalculate(self)
            cost = cld.cost
            costs.append(cost)
            if cost < min_cost:
                min_cost = cost
                best_calendar = cld

        return best_calendar, costs

    # assume availability is in alphabetical order
    def make_calendar(self, year, month,
                      availability,
                      off_days=None,
                      assignments=None,
                      n=2000,
                      force_assignments=False,
                      momentum=0.2):
        '''
        Parameters
        ----------
        year : int
            year of the calendar
        month : int
            month of the calendar
        availability : str or 2D binary array
            if str, must be the path to a csv file containing the 
            availabilities of all RAs on staff. Rows are assumed to be the RAs
            in alphabetical order of their name. Each column is a day, 0 means 
            unavailable, 1 means available.
            if array, same convention as above applies
        off_days : list of int
            any days which should not have an on-call shift for the month
        assignments : dict with str keys and list of int values
            any forced assignments to be added before calculating the rest of
            the calendar
            eg. {'Katrina Doe' : [3, 10, 23], 'Sergio Dae : [5]'}
        n : int
            number of calendars to calculate
        force_assignments : bool
            will automatically assign RAs to days that have no availability
            with the goal of balancing out total points
            
        Returns
        -------
        calendar : Calendar object
            contains an array of RA objects representing the on-call
            assignments for the given month and year
            includes functions to alter assignments and save calendar as
            readable csv file
        '''
        # ensure RAs are sorted by name
        self.RAs = Staff.sort_RAs(self.RAs)
        # set defaults
        if off_days is None:
            off_days = []
        if assignments is None:
            assignments = {}
        # validate the parameters
        availability, off_day_indices, assignments = self.prep_av(
            year, month, availability, off_days, assignments)
        # calculate how many days available is considered a low availability
        RAs = self.RAs
        curr_points = self.get_points()
        print("\nCalculating calendar... please wait")
        best_calendar, costs = self.generate_calendar(year
                                                      , month
                                                      , availability
                                                      , off_day_indices
                                                      , assignments
                                                      , n
                                                      , force_assignments
                                                      , momentum)
        # return all RA values to what they were
        for i, RA in enumerate(RAs):
            RA.is_low_avail = False
            RA.is_overloaded = False
            RA.days_on_call = 0
            RA.weekends_on_call = 0
            RA.points = curr_points[i]
        print(f"The mean cost for {n} generated calendars was"
              f" {np.mean(costs) : .2f}."
              f" The returned calendar (#{np.argmin(costs)}/{len(costs)}) has a"
              f" cost of {np.min(costs) : .2f}.")
        forced_assignments = best_calendar.forced_availability_cost
        if forced_assignments != 0:
            eprint("[WARNING]: The returned calendar has made"
                  f" {forced_assignments} forced assignment(s).\n"
                  "i.e. the algorithm did not find a way to preserve fairness"
                  " while respecting the given availability.")
        return best_calendar
    
    def apply_calendar(self, calendar, undo=False):
        weekends = self.weekends
        weekend_value = self.weekend_value
        if undo is True:
            coeff = -1
        else:
            coeff = 1
        for day_index, RA in enumerate(calendar.calendar):
            day_week = cal.weekday(
                calendar.year, calendar.month, day_index + 1)
            if RA is None:
                continue
            else:
                try:
                    RA_index = Staff.get_index(self.RAs, RA)
                # if RA no longer exists continue in the loop
                except:
                    continue
            if day_week in weekends:
                self.RAs[RA_index].add_points(coeff * weekend_value)
            else:
                self.RAs[RA_index].add_points(coeff)
       
    # expects list of RA objects and a target RA object
    # searches name-wise
    @staticmethod
    def get_index(list_RAs, target):
        names = np.asarray([RA.name for RA in list_RAs])
        try:
            # if not a string, get the name of the RA
            if not isinstance(target, str):
                name = target.name
            else:
                name = target
            # expect names to be unique
            RA_index = np.flatnonzero(names == name)[0]
            return RA_index
        except IndexError:
            raise ValueError(f"'{name}' not found")
        except:
            raise

# a single RA/employee. has personal details such as availability,
# number of days on call, etc, that is only of concern to
# the Staff during calendar creation
class RA:

    def __init__(self, name, points=0):
        self._name = name
        self._points = points

        # to be used for calculating calendars ONLY, helps organization
        self.availability = None
        self.is_low_avail = False
        self.is_overloaded = False
        self.days_on_call = 0
        self.weekends_on_call = 0

    def __str__(self):
        return 'Name: ' + self.name + ' Points: ' + str(self.points)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if not re.fullmatch(r'([A-Za-z]+ ?)+', name):
            raise ValueError(f"Invalid Name: '{name}'"
                             " name must only contain letters")
        else:
            self._name = name

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if points < 0:
            raise ValueError("Points must be non-negative")
        else:
            self._points = points

    def add_points(self, points):
        self._points += points
        
class CalRA(RA):
    def __init__(self):
        self.availability = None
        self.is_low_avail = False
        self.is_overloaded = False
        self.days_on_call = 0
        self.weekends_on_call = 0

# the idea is for the calendar to keep a snapshot of the state of the Staff;
# the settings, RAs, and availability used when the calendar was made are
# saved, allowing for only minor tweaks of the calendar
# this is to 
class Calendar:

    # the calendar uses many attributes from the staff and RAs that were
    # used to make it
    # however, it does not take in the entire staff as an attribute because
    # the staff can change in the future, and I don't want the calendar
    # to reflect those changes
    def __init__(self, year, month, RAs, availability):
        # immutable upon creation
        self.year = year
        self.month = month
        self.number_days = cal.monthrange(year, month)[1]
        # the order of this array is assumed for all arrays
        self.RAs = RAs
        self.availability = availability
        # copied from the staff at the time of creation
        self.weekends = [4,5]
        self.weekend_value = 2
        
        # mutable through 'switch' or 'apply'
        self.calendar = np.full(self.number_days, None, dtype='object')
        self.been_applied = False
        # array of points of the RAs for this calendar
        self.points = None
        # array of number of days on call for each RA
        self.days_on_call = None
        # array of number of weekends on call for each RA
        self.weekends_on_call = None
        # array of day indices that do not respect availability
        self.forced_availability = None
        # measure of how good the calendar is relative to others
        self.cost = 0
        # cost of not respecting availability
        self.forced_availability_cost = 0
        # cost of having RAs be on call multiple days in a row
        self.adjacent_cost = 0
        

    def __str__(self):
        return self.print_calendar_representation(give=True)

    # calculate a cost for RAs being on-call on adjacent or near-adjacent days
    def calculate_adjacent_cost(self):
        cost = 0
        for i in range(self.number_days - 2):
            current_RA = self.calendar[i]
            if current_RA is None:
                continue
            next_day_RA = self.calendar[i + 1]
            overmorrow_RA = self.calendar[i + 2]
            # if RA is on-call three days in a row, really high cost
            if current_RA is overmorrow_RA and current_RA is next_day_RA:
                cost += 6
            # some cost if RA is on-call two days in a row
            elif current_RA is next_day_RA:
                cost += 2
            # small cost if RA is on-call the day after tomorrow
            elif current_RA is overmorrow_RA:
                cost += 1
        return cost

    # calculate a cost for RAs that are made on-call when they are unavailable
    def calculate_forced_availability_cost(self):
        cost = 0
        fa = []
        for day_index, RA in enumerate(self.calendar):
            if RA is None:
                continue
            else:
                RA_index = Staff.get_index(self.RAs, RA)
                if self.availability[RA_index, day_index] == 0:
                    fa.append(day_index)
                    cost += 1
        self.forced_availability = fa
        return cost

    # an array of points for this calendar
    def calendar_points(self):
        days_on_call = np.asarray(self.days_on_call)
        weekends_on_call = np.asarray(self.weekends_on_call)
        weekend_value = self.weekend_value

        return (days_on_call - weekends_on_call) + weekend_value * weekends_on_call

    def calculate_cost(self):
        '''
        A function that attempts to quantify how good a calendar is by
        taking into account the point distribution, number of days on call,
        adjacent days on call, and more. Calendars with unfavorable
        characteristics will have a relatively higher cost compared to a 
        more balanced calendar.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        cost : float
            a number that represents how good a calendar is
        
        '''

        weights = np.asarray([20, 15, 15, 4, 2, 2])
        costs = np.asarray([
            np.std(self.points),  # std of total point distribution
            np.std(self.calendar_points()),  # std of calendar point distribution
            self.forced_availability_cost,  # number of forced assignments
            np.std(self.days_on_call),  # std of number of days on call
            np.std(self.weekends_on_call),  # std of number of weekends on call
            self.adjacent_cost])  # number of adjacent days
        cost = np.dot(weights, costs)
        return cost
        
    def recalculate(self, staff):
        self.weekends = staff.weekends
        self.weekend_value = staff.weekend_value
        year = self.year
        month = self.month
        weekends = self.weekends
        weekend_value = self.weekend_value
        # takes care of any new RAs that may have been added after the
        # calendar was made
        cal_names = [RA.name for RA in self.RAs]
        staff_name_to_RA = dict(zip(staff.get_names(), staff.RAs))
        RAs = self.RAs
        for i, name in enumerate(staff_name_to_RA.keys()):
            if name not in cal_names:
                self.RAs = np.append(self.RAs, staff_name_to_RA[name])
                try:
                    self.availability = np.insert(
                        self.availability, i, True, axis = 0)
                except:
                    self.availability = np.insert(
                        self.availability, self.availability.shape[0], True, axis = 0)
        self.RAs = Staff.sort_RAs(self.RAs)
        RAs = self.RAs
        cal_names = [RA.name for RA in self.RAs]
    
        days_on_call = np.zeros(RAs.size, dtype='int')
        weekends_on_call = np.zeros(RAs.size, dtype='int')
        points = np.zeros(RAs.size)
        for day_index, RA in enumerate(self.calendar):
            if RA is None:
                continue
            RA_index = Staff.get_index(RAs, RA)
            day_week = cal.weekday(year, month, day_index + 1)
            if day_week in weekends:
                weekends_on_call[RA_index] += 1
                points_to_add = weekend_value
            else:
                points_to_add = 1
            days_on_call[RA_index] += 1
            points[RA_index] += points_to_add
        self.points = points
        self.days_on_call = days_on_call
        self.weekends_on_call = weekends_on_call
        self.adjacent_cost = self.calculate_adjacent_cost()
        self.forced_availability_cost = self.calculate_forced_availability_cost()
        self.cost = self.calculate_cost()

    def print_cost_breakdown(self):
        result = 'Calendar cost breakdown\n'
        result += '-----------------------\n'
        result += f' {"20*Std of point distribution: " : <29}{20 * np.std(self.points) : >10.2f}\n'
        result += f'+ {"15*Calendar Point std: " : <29}{15 * np.std(self.calendar_points()) : >10.2f}\n'
        result += f'+ {"15*Forced Availability: " : <29}{15 * self.forced_availability_cost : >10.2f}\n'
        result += f'+ {"4*Std of days on call: " : <29}{4 * np.std(self.days_on_call) : >10.2f}\n'
        result += f'+ {"2*Std of weekends on call: " : <29}{2 * np.std(self.weekends_on_call) : >10.2f}\n'
        result += f'+ {"2*Adjacent days on call: " : <29}{2 * self.adjacent_cost : >10.2f}\n'
        result += '-----------------------------------------\n'
        result += f'  {"Calendar cost:" : <29}{self.cost : >10.2f}'

        print(result)

    def print_forced_availability(self):
        '''Print any RAs that were made on call on a day they
        stated they were unavailable
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''

        if self.forced_availability_cost == 0:
            return

        for RA, row in zip(self.RAs, self.availability):
            RA.availability = row

        result = 'RAs forced on call when unavailable\n'
        result += '-----------------------------------\n'
        for i, RA in enumerate(self.calendar):
            if RA is None:
                continue
            elif RA.availability[i] == 0:
                day_name = cal.day_name[
                    cal.weekday(self.year, self.month, i + 1)]
                result += f"{RA.name} on {i + 1} ({day_name})\n"
        print(result)
    
    def assign(self, RA, day):
        RA_index = Staff.get_index(self.RAs, RA)
        chosen_RA = self.RAs[RA_index]
        self.calendar[day - 1] = chosen_RA
    
    def switch_days(self, day1, day2):
        ndays = self.number_days
        if min(day1,day2) < 1 or max(day1,day2) > ndays:
            raise ValueError(f"Days must be between 1 and {ndays}")
            return
        day1_RA = self.calendar[day1 - 1]
        day2_RA = self.calendar[day2 - 1]
        self.calendar[day1 - 1] = day2_RA
        self.calendar[day2 - 1] = day1_RA

    # returns an array of strings in standard calendar format
    def calendar_format(self):
        first_day = (cal.weekday(self.year, self.month, 1) + 1) % 7

        # check how many rows are needed for the calendar
        if first_day + self.number_days > 35:
            num_weeks = 6
        else:
            num_weeks = 5

        calendar = np.full((num_weeks + 1, 7), '', dtype='U23')

        # weekday name header starting from Sunday
        for i in range(7):
            calendar[0, i] = f"{cal.day_name[(i - 1) % 7]}"

        day_counter = 0
        # do the first week
        for day in range(first_day, 7):
            RA = self.calendar[day_counter]
            if RA is None:
                calendar[1, day] = f'{day_counter + 1} None'
            else:
                calendar[1, day] = f"{day_counter + 1} {RA.name}"
            day_counter += 1

        # do the rest of the weeks
        for week in range(2, num_weeks + 1):
            for day in range(7):
                RA = self.calendar[day_counter]
                if RA is None:
                    calendar[week, day] = f'{day_counter + 1} None'
                else:
                    calendar[week, day] = f"{day_counter + 1} {RA.name}"
                day_counter += 1
                if day_counter >= self.number_days:
                    break

        return calendar

    def save_calendar_representation_csv(self, path):
        '''Saves the calendar as a human-readable .csv file that can be
        opened in excel
        
        Parameters 
        ----------
        filename : str
            name of the file to be saved
        
        Returns
        -------
        None
        '''

        calendar = self.calendar_format()
        np.savetxt(path, calendar, delimiter=',', fmt='%s')

    def print_calendar_representation(self, give=False):
        '''String representation of the calendar
        
        Parameters 
        ----------
        give : bool
            if True, returns the string representation
            if False, prints the string representation
        
        Returns
        -------
        result : str
            only if give is True
        '''
        # for formatting
        name_lens = list(map(len, [RA.name for RA in self.RAs]))
        max_len = max(10, min(23, max(name_lens)))

        calendar = self.calendar_format()

        result = ''
        day_index = -1*(7 + np.flatnonzero(calendar[1])[0])
        for row in calendar:
            for entry in row:
                if day_index in self.forced_availability:
                    result += (f"{Fore.LIGHTRED_EX}{entry : <{max_len + 5}}"
                               f"{Style.RESET_ALL}")
                else:
                    result += f"{entry : <{max_len + 5}}"
                day_index += 1
            result += '\n'

        if give is True:
            return result
        else:
            print(result)
            
    # tabular breakdown of distribution of points and day assignments
    def info(self):
        weight = self.weekend_value
        names = np.sort([RA.name for RA in self.RAs])
        lens_of_names = list(map(len, names))
        # find the longest name to know how to format the result
        max_len = max(16, max(lens_of_names)) + 1
        days_on_call = np.asarray(self.days_on_call)
        weekends_on_call = np.asarray(self.weekends_on_call)

        points_from_this = ((days_on_call - weekends_on_call) 
                            + weight * (weekends_on_call))

        result = f"{'Name' : <{max_len}}{'Wkdays On-call' : <{max_len}}"
        result += f"{'Wknds On-call' : <{max_len}}{'Cal Points' : ^{max_len}}"
        result += f"{'Total Points' : ^{max_len}}\n"
        for i in range(max_len * 5):
            result += '-'
        result += '\n'
        for i, RA in enumerate(self.RAs):
            result += f"{RA.name : <{max_len}}"
            result += f"{days_on_call[i] - weekends_on_call[i] : ^{max_len}.0f}"
            result += f"{weekends_on_call[i] : ^{max_len}.0f}"
            result += f"{points_from_this[i] : ^{max_len}.1f}"
            result += f"{RA.points : ^{max_len}.1f}\n"
        print(result)


class StaffCmd(cmd.Cmd):
    global staff
    global my_cal
    global availability
    global quotes
    global staff_path
    global common_path
    global last_cal_path
    global avail_path
    global saved_path
    
    
    init()

    app_dir = os.path.abspath(os.path.dirname(__file__))
    # absolute paths to all needed directories
    common_path = os.path.abspath(os.path.join(app_dir, 'calgen_common'))
    staff_path = os.path.abspath(os.path.join(common_path, 'my_staff.pkl'))
    last_cal_path = os.path.abspath(os.path.join(common_path, 'last_cal.pkl'))
    avail_path = os.path.abspath(os.path.join(app_dir, 'availabilities'))
    saved_path = os.path.abspath(os.path.join(app_dir, 'saved_calendars'))
    os.chdir(app_dir)
    all_dirs = os.listdir()
    # create necessary folders if they don't exist
    # folder to contain the Staff, Calendar objects as well
    # as the last loaded availability
    if 'calgen_common' not in all_dirs:
        os.mkdir('calgen_common')
    # where saved calendars go
    if 'saved_calendars' not in all_dirs:
        os.mkdir('saved_calendars')
    # where the user stores the availabilities
    if 'availabilities' not in all_dirs:
        os.mkdir('availabilities')
    # get contents of common folder
    dir_contents = [x for x in os.listdir('calgen_common')]
    try:
        with open(os.path.join(common_path,"quotes.txt"), 'r') as f:
            content = f.read()
            quotes = np.asarray(content.split('\n'))
    except:
        quotes = ['']
    # open the staff
    # if there is no staff
    if 'my_staff.pkl' not in dir_contents:
        while True:
            try:
                print("Welcome to calgen!")
                time.sleep(1)
                print("No saved staff found."
                      " Would you like to create one? y/n")
                key_press = input('calgen: ')
                if key_press == 'y':
                    new_staff_name = input('Name of staff: ')
                    names_of_RAs = input("Names of RAs separated by ',': ")
                    names_of_RAs_list = re.split(r',\s*', names_of_RAs)
                    names_of_RAs_list = list(
                        map(str.strip, names_of_RAs_list))
                    for name in names_of_RAs_list:
                        if not re.fullmatch(r'([A-Za-z]+ ?)+', name):
                            raise ValueError(
                                f"Invalid Name: '{name}'"
                                " name must only contain letters")
                    staff = Staff(new_staff_name, names_of_RAs_list)
                    print("Does this look good? y/n")
                    time.sleep(1)
                    print(staff)
                    key_press = input('calgen: ')
                    if key_press == 'y':
                        print(f"New staff {staff.staff_name}"
                              " created and saved!")
                        with open(staff_path, 'wb') as f:
                            pickle.dump(staff, f)
                        time.sleep(2)
                        break
                    elif key_press == 'n':
                        pass
                    else:
                        eprint(f"Invalid input: '{key_press}'")
                        pass
                elif key_press == 'n':
                    print("quitting...")
                    sys.exit()
                else:
                    eprint("Invalid input: '{key_press}'")
                    pass
            except SystemExit:
                sys.exit()
            except Exception as e:
                eprint(repr(e))
    else:
        try:
            with open(staff_path, 'rb') as f:
                staff = pickle.load(f)
        except Exception as e:
            eprint(repr(e))
    try:
        if 'last_cal.pkl' in dir_contents:
            with open(last_cal_path, 'rb') as f:
                my_cal = pickle.load(f)
        if 'last_avail.csv' in dir_contents:
            path = os.path.join(common_path, "last_avail.csv")
            availability = np.genfromtxt(path, delimiter=',')
        else:
            availability = None
            av_warning = (f"\n{Fore.LIGHTRED_EX}[WARNING]: No availability loaded."
                  " Use function 'ldav'"
                  " before \nattempting to use the 'mkcal' utility"
                  f"{Style.RESET_ALL}")
    except Exception as e:
        eprint(repr(e))
        
    intro = f"""{Fore.LIGHTBLUE_EX}          |                   
,---.,---.|    ,---.,---.,---.
|    ,---||    |   ||---'|   |
`---'`---^`---'`---|`---'`   |
               `---'         '  v1.0 {Style.RESET_ALL}"""
    if availability is None:
        intro += f"{av_warning}"   
    intro += f"""
Welcome {staff.staff_name} SRA!
'help' for help
'exit' to exit"""
    

    prompt = 'calgen: '
    
    # * more detailed documentation for the multiple 
    # * 'set' command handler commands
    def help_set_pts(self):
        print('''
        Usage: set pts <name> <points>
        
        Changes the points of an RA.
        
        Parameters
        ----------
        name : str
            name of the RA to modify
        points : int
            number of points to set
            
        eg. calgen: set pts Jane Doe 10
            Jane Doe' now has 10.0 points
        ''')
        
    def help_set_name(self):
        print('''
        Usage: set name <old_name> , <new_name>
                                   ^ mind the comma
        Changes the name of an RA.
        
        Parameters
        ----------
        old_name : str
            current name of the RA whose name to change
        new_name : str
            new name for the chosen RA
            
        eg. calgen: set name John D , John Doe
            Changed name 'John D' to 'John Doe'
        ''')
        
    def help_set_ends(self):
        print('''
        Usage: set ends <days>
        
        Set which days count as weekends.
        
        Parameters
        ----------
        days : int 0-6 inclusive separated by spaces or ','
            the day(s) you wish to count as weekends. The number 0 corresponds
            to Monday, 1 Tuesday, ..., 6 Sunday
            
        eg. calgen: set ends 4,5
            Weekends changed to Friday Saturday
        ''')

        

    def help_set_factor(self):
        print('''
        Usage: set factor <value>
        
        Set how many points a weekend is worth. The calendar generating
        algorithm was made and tested with a weekend value of 2.
        
        Parameters
        ----------
        value : float or int
            the number of points a weekend will be worth
            all other days are worth 1 point
            
        eg. calgen: set factor 1.5
            Weekend value changed to 1.5
        ''')
        
    # * Methods to change attributes, not directly user callable

    def set_pts(self, arg):
        name = ''
        for t in arg.split():
            try:
                points = float(t)
            except ValueError:
                name += f"{t} "
        name = name.strip()

        if name == '':
            self.help_set_points()
            return

        try:
            name_to_RAobject = dict(zip(staff.get_names(), staff.RAs))
            chosen_RA = name_to_RAobject[name]
            chosen_RA.points = points
            print(f"'{chosen_RA.name}' now has {points} points")
            self.save()
        except KeyError:
            e = KeyError(f"Given name '{name}' does not exist.")
            eprint(repr(e))
        except UnboundLocalError:
            print("Usage: set points <name> <points>")
        except Exception as e:
            eprint(repr(e))

    def set_name(self, arg):

        try: 
            args = re.split(r'\s*,\s*', arg.strip())
            if len(args) != 2:
                self.help_set_name()
                return
            old_name, new_name = args[0], args[1]
            name_to_RA = dict(zip(staff.get_names(), staff.RAs))
            if new_name in name_to_RA.keys():
                raise ValueError(f"'{new_name}' name already taken")
            chosen_RA = name_to_RA[old_name]
            chosen_RA.name = new_name
            print(f"Changed name '{old_name}' to '{new_name}'")
            self.save()
        except KeyError:
            e = KeyError(f"Given name '{old_name}' does not exist.")
            eprint(repr(e))
        except ValueError as e:
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))

    def set_factor(self, arg):
        if arg == '':
            self.help_set_factor()
            return
        try:
            value = float(arg)
            if value < 1:
                raise ValueError("Weekend value must be >= 1")
            staff.weekend_value = value
            self.save()
        except Exception as e:
            eprint(repr(e))
            print("Usage: set weekend_value <value>")

    def set_ends(self, arg):

        if arg == '':
            self.help_set_ends()
            return
        
        if re.fullmatch(r'(\d\s*,?\s*)+', arg):
            try:
                staff.weekends = StaffCmd.parse_int(arg)
                self.save()
            except IndexError:
                e = IndexError("Weekends must be int from 0 to 6 inclusive")
                eprint(repr(e))
            except Exception as e:
                eprint(repr(e))
        else:
            print("Usage: set weekends <days>")

    # * begin user callable commands

    # quick overview of the staff
    def do_show(self, arg):
        '''
        Usage: show [options]
        
        Prints a table of names and points of RAs on your staff as well
        as the current settings.
        
        Options
        -------
        -n    --names : sorts by name, default value
        -p    --points : sorts by points
        -pr    --pointsr : sorts by points reversed
        '''
        arg = arg.strip()
        
        if arg == '-p' or arg == '--points':
            sort_flag = 'points'
        elif arg == '-pr' or arg == '--pointsr':
            sort_flag = 'pointsr'
        else:
            sort_flag = 'names'

        weekends = ''
        if len(staff.weekends) == 0:
            weekends += 'None'
        else:
            for day in staff.weekends:
                weekends += f"{cal.day_name[day]} "

        print("")
        print(f"{Fore.LIGHTBLUE_EX}{staff.staff_name}{Style.RESET_ALL}")
        print(f"The following days count as weekends: {weekends}")
        print(f"Weekends (weekdays) are worth {staff.weekend_value} (1) points. ")
        staff.print_RAs(sort=sort_flag)

    def do_add(self, arg):
        '''
        Usage: add <name> [points]
        
        Add a new RA to your staff. The RA will automatically be given the
        average number of points of RAs in your current staff. Can override
        by providing an integer after the name argument.
        
        Parameters
        ----------
        name : str
            name of the RA to be added
        (optional) points : int
            points to be given to the RA
        '''
        # parse the arguments
        args = re.split(r"\s+", arg.strip())
        if args == ['']:
            self.do_help("add")
            return
        try:
            name = ' '.join(args[:-1])
            points = int(args[-1])
        except ValueError:
            name = ' '.join(args)
            points = None
        except Exception as e:
            eprint(repr(e))
            
        try:
            # ensure name has no padding whitespace
            if not re.fullmatch(r'([A-Za-z]+ ?)+', name):
                raise ValueError(
                    f"Invalid Name: '{name}' name must only contain letters")
            staff.add_RAs(name, points)
            print(f"Added RA '{name}' to staff")
            self.save()
        except ValueError as e:
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))
    
    def do_remove(self, arg):
        '''
        Usage: remove <name>
        
        Removes an RA by name from your staff. Cannot be undone.
        
        Parameters
        ----------
        name : str
            name of the RA to be removed
        '''
        
        name = arg

        if name == '':
            self.do_help('remove')
            return
        
        try:
            staff.delete_RAs(name)
            print(f"Removed RA '{name}' from staff")
            self.save()
        except ValueError as e:
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))
    
    
    # command handler for all possible set commands
    # made to reduce the number of commands the user has to remember
    def do_set(self, arg):
        '''
        Usage: set <attribute> <arguments>
        
        Changes an attribute belonging to an RA or to the Staff.
        
        Parameters
        ----------
        attribute : str in (name, pts, ends, factor)
            the property to change
        arguments : varies
            the arguments required to change the chosen attribute
            
        <attribute> : <arguments> pairs
        -------------------------------
        name : <old_name> , <new_name>
        pts : <name> <points>
        ends : <days>
        factor : <value>
        
        type 'help set_<attribute>' for more help
        '''
        try:
            attr = arg.split()[0]
        except:
            self.do_help('set')
            return

        args = ' '.join(arg.split()[1:])
        try:
            if attr == 'pts':
                self.set_pts(args)
            elif attr == 'name':
                self.set_name(args)
            elif attr == 'ends':
                self.set_ends(args)
            elif attr == 'factor':
                self.set_factor(args)
            else:
                eprint(f"Invalid Attribute: '{attr}'")
        except Exception as e:
            eprint(repr(e))
        
            
    # better than having the availability passed as an argument to the
    # 'makecal' function. user can make many calendars with less typing
    def do_load(self, arg):
        '''
        Usage: load <filename>
        
        Loads a csv from the 'availabilities' folder representing the 
        availability of your staff. The program assumes that each row
        in your csv is a staff member, sorted alphabetically as in 'show'.
        Each column represents a day. The csv must be binary, only 0 or 1 
        should be in each entry, no headers. You can make your own csv with the
        'makeav' function or make one in a spreadsheet app and save
        as csv.
        
        Parameters
        ----------
        filename : str
            the name of the csv you are trying to load
        '''
        global availability

        try:
            path1 = os.path.join(avail_path, arg)
            path2 = os.path.join(common_path, "last_avail.csv")
            availability = np.genfromtxt(path1, delimiter=',', dtype = 'int')
            np.savetxt(path2, availability, delimiter=',')
            print(f"Successfully loaded {arg}")
        except OSError as e:
            eprint(repr(e))         
        except Exception as e:
            eprint(repr(e))
    
    # show all available arguments for 'load' function
    # to troubleshoot for things like missing file extensions
    def do_showav(self, arg):
        '''
        Prints the contents of the 'availabilities' folder; the folder that
        'load' uses. These are the possible arguments for
        'load'.
        '''
        avs = sorted(os.listdir(avail_path))
        if len(avs) == 0:
            print(f"{Fore.LIGHTRED_EX}No files{Style.RESET_ALL}")
            return
        print(f"{Fore.LIGHTBLUE_EX}")
        for name in avs:
            print(name)
        print(f"{Style.RESET_ALL}")
     
        
    # user can make their own availability csv without having to 
    # manually enter 1s and 0s into an excel file
    def do_makeav(self, arg):
        '''
        Usage: makeav <year> <month>
        
        A simple utility to make an availability csv for calgen to read.
        The utility will loop through all RAs on staff and ask one by one for
        input about the days that the member is available.
        
        Parameters
        ----------
        year : int YYYY format
            year for the availability
        month : int 1-12 inclusive
            month for the availability
            
        eg. Available days: 1,4,5:8,15,20:22
            means available on [1,4,5,6,7,8,15,20,21,22]
        '''

        try:
            args = re.split(r'[,\s]+', arg)
            if len(args) < 2 or len(args) > 2:
                self.do_help("makeav")
                return
            year = int(args[0])
            month = int(args[1])
            if len(str(year)) != 4:
                raise ValueError("Invalid Format: YYYY required")
            num_days = cal.monthrange(year, month)[1]
        except Exception as e:
            eprint(repr(e))
            return

        print(f"{Fore.LIGHTBLUE_EX}")
        print("Please enter day numbers/ranges separated by ','")
        print("x:y denotes days from x to y inclusive")
        print(f"Type 'exit' to exit the utility whenever {Style.RESET_ALL}")
        av = np.zeros((staff.RAs.size, num_days), dtype = 'int')
        for i, name in enumerate(staff.get_names()):
            while True:
                try:
                    print("\nEnter the available days for"
                          f" {Fore.LIGHTBLUE_EX}{name}{Style.RESET_ALL}")
                    key_press = input("Available days: ")
                    if key_press == 'exit':
                        return
                    else:
                        days = StaffCmd.parse_int(key_press)
                    if (np.any(np.asarray(days) > 31)
                            or np.any(np.asarray(days) < 1)):
                        raise ValueError(
                            f"Day numbers must be between 1 and {num_days}")
                    days = np.asarray(days)
                    np.put(av[i], days - 1, 1)
                    break
                except ValueError as e:
                    eprint(repr(e))
                except Exception as e:
                    eprint(repr(e))

        while True:
            try:
                print(f"\n{Fore.LIGHTBLUE_EX}"
                      f"Enter filename: {Style.RESET_ALL}", end = '')
                filename = input()
                if filename == 'exit':
                    return
                if filename[-4:].lower() != '.csv':
                    filename += '.csv'
                path = os.path.join(avail_path, filename)
                np.savetxt(path, av, delimiter=',')
                print("Availability csv saved in "
                      f"{Fore.LIGHTBLUE_EX}{path}{Style.RESET_ALL}")
                break
            except OSError as e:
                eprint(repr(e))
            except Exception as e:
                eprint(repr(e))
    
    # primary function of the program
    def do_makecal(self, arg):
        '''
        Usage: makecal <year> <month> [options]
        
        Generates a calendar with the loaded availability. Will never assign
        an RA to a day they mark as unavailable. This can result in some days
        being marked with None. Calendar functions include 'showcal', 
        'save', and 'switch'. If unhappy with the generated calendar, you
        can make a new one by calling the function again or use the function
        'switch' to make changes to the calendar.
        
        Parameters
        ----------
        year : int YYYY format
            year for the calendar
        month : int
            month for the calendar
            
        Options
        -------
        --offdays : prompts user to enter days not to be assigned. If the 
                loaded availability has any days with no RAs available,
                no one will be assigned to that day.
        --assign : will prompt the user to enter any assignments to make.
                Can be used to guarantee that certain RAs get paired with
                certain days on call.
        '''
        global availability
        global my_cal
        global staff
        
        flags_used = False
        off_days = []
        assignments = {}

        try:
            args = re.split(r'[ ,]+', arg)
            if len(args) < 2:
                self.do_help("makecal")
                return
            if availability is None:
                raise ValueError(
                    "No availability loaded. Use 'load' first")
            year = int(args[0])
            month = int(args[1])
            if len(str(year)) != 4:
                raise ValueError("Invalid Format: YYYY required")
            num_days = cal.monthrange(year, month)[1]
            valid_flags = ['--offdays', '--assign']
            flags = np.unique(args[2:]) if len(arg) > 2 else []
            for flag in flags:
                if flag not in valid_flags:
                    self.do_help("makecal")
                    raise  ValueError(f"Invalid option: {flag}")
                flags_used = True
        except Exception as e:
            eprint(repr(e))
            return
        
        if flags_used is True:
            print(f"{Fore.LIGHTBLUE_EX}")
            print("Please enter day numbers/ranges separated by ','")
            print("x:y denotes days from x to y inclusive")
            print(f"{Style.RESET_ALL}")

        # flag handling
        if '--offdays' in flags:
            while True:
                try:
                    key_press = input("Days off: ")
                    off_days = StaffCmd.parse_int(key_press)
                    if (np.any(np.asarray(off_days) > num_days)
                            or np.any(np.asarray(off_days) < 1)):
                        raise ValueError(
                            f"Day numbers must be between 1 and {num_days}")
                    print(f"The days {off_days} will not be assigned")
                    break
                except ValueError as e:
                    eprint(repr(e))
            time.sleep(1)
        if '--assign' in flags:
            flags_used = True
            all_names = []
            all_days = []
            print("{Fore.LIGHTBLUE_EX}")
            print("You will make assignments one RA at a time")
            print("Please enter day numbers separated by ','")
            print("Type 'names' to see all available names")
            print("Type 'done' into name field when finished"
                    "{Style.RESET_ALL}")
            while True:
                try:
                    name = input("RA name: ")
                    if name == 'done':
                        break
                    elif name == 'names':
                        print(staff.get_names())
                        continue
                    elif name in all_names:
                        raise ValueError(f"RA {name} has already"
                                            " been assigned") 
                    elif name not in staff.get_names():
                        raise ValueError(f"Name '{name}' does not"
                                            " exist in staff")
                    days = list(np.unique(
                        StaffCmd.parse_int(
                            input("Days to assign: "))))
                    for day in days:
                        if day in np.ravel(all_days):
                            raise ValueError(
                                f"Day {day} has already been assigned")
                        if day in off_days:
                            raise ValueError(
                                f"Cannot assign an RA to an off day: {day}")
                    if (np.any(np.asarray(days) > 31)
                            or np.any(np.asarray(days) < 1)):
                        raise ValueError(
                            f"Day numbers must be between 1 and {num_days}")
                    print(f"RA {name} assigned to days {days}")
                    all_names.append(name)
                    all_days.append(days)
                except Exception as e:
                    eprint(repr(e))
            assignments = dict(zip(all_names, all_days))
            print(f"Assignments set to {assignments}")

        if flags_used is True:
            while True:
                try:
                    my_cal = staff.make_calendar(year
                                                 , month
                                                 , availability
                                                 , off_days
                                                 , assignments)
                    print("")
                    print(my_cal)
                    my_cal.info()
                except Exception as e:
                    eprint(repr(e))
                    return
                print("If you are unhappy with this calendar, consider "
                      "running the function again")
                print(f"{Fore.LIGHTBLUE_EX}"
                      "Generate new calendar with same settings? y/n"
                      f"{Style.RESET_ALL}")
                key_press = input("calgen: ")
                if key_press == 'y':
                    continue
                elif key_press == 'n':
                    break
                else:
                    break
        else:
            try:
                my_cal = staff.make_calendar(year
                                             , month
                                             , availability
                                             , off_days
                                             , assignments)

                print("")
                print(my_cal)
                my_cal.info()
            except Exception as e:
                eprint(repr(e))
                return
            print("If you are unhappy with this calendar, consider running "
                  "the function again")
        print("To save this calendar to your computer,"
              " call 'save'")
        print("To apply this calendar's points to your staff,"
              " call 'apply'")
        path = os.path.join(common_path,"last_avail.csv")
        np.savetxt(path, availability, delimiter=',')
        with open(last_cal_path, 'wb') as f:
            pickle.dump(my_cal, f)

    # allows for small changes to a calendar
    def do_switch(self, arg):
        '''
        Usage: switch <day1> <day2>
        
        Switches two RAs in the last generated calendar. The availability
        used to make the calendar is ignored.
        
        Parameters
        ----------
        day1 : int
        day2 : int
            the days to be switched in the calendar
        '''
        try:
            days = StaffCmd.parse_int(arg)
            if len(days) != 2:
                self.do_help("switch")
            my_cal.switch_days(days[0], days[1])
            my_cal.recalculate(staff)
            self.save()
            print("Switch successful")
        except NameError:
            e = NameError("You have not generated any calendars")
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))
            
    def do_assign(self, arg):
        '''
        Usage: assign <name> <day>
        
        Parameters
        ----------
        name : str
            name of RA to assign
        day : int
            day to assign the RA to
        
        eg. assign Jane Doe 12
        '''
        # parse
        name = ''
        day = None
        for t in arg.split():
            try:
                day = int(t)
            except ValueError:
                name += f"{t} "
        name = name.strip()

        if name == '' or day is None:
            self.do_help("assign")
            return
    
        try:
            my_cal.assign(name, day)
            my_cal.recalculate(staff)
            day_name = cal.day_name[
                cal.weekday(my_cal.year, my_cal.month, day)]
            if str(day)[-1] == '1':
                ending = "st"
            elif str(day)[-1] == '2':
                ending = 'nd'
            elif str(day)[-1] == '3':
                ending = 'rd'
            else:
                ending = 'th'
            print(f"Assigned '{name}' to {day_name} the {day}{ending}")
        except NameError:
            e = NameError("You have not generated any calendars")
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))

    def do_showcal(self, arg):
        '''
        Shows the last generated calendar in the command line
        '''
        try:
            print("\nConflicts with availability in"
                  f" {Fore.LIGHTRED_EX}RED{Style.RESET_ALL}\n"
                  f"{my_cal}") 
            my_cal.info()
        except NameError:
            e = NameError("You have not generated any calendars")
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))

    def do_apply(self, arg):
        '''
        Applies the last generated calendar to the staff. That is, all of
        the calculated points from the last calendar will be added to the
        RAs
        '''
        try:
            if my_cal.been_applied is False:
                staff.apply_calendar(my_cal)
                my_cal.been_applied = True
                print("The last generated calendar was applied to your staff.")
                print("Call 'show' to see updated points")
                # save as the last calendar that was applied
                self.save()
            else:
                raise ValueError(
                    "Cannot apply a calendar twice. The last generated"
                    " calendar has already been applied to your staff before")
        except NameError:
            e = NameError("No saved calendar found")
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))

    def do_undo_apply(self, arg):
        '''
        Unapplies the last generated calendar. That is,
        all points from the last applied calendar are subtracted from all RAs.
        Undo depth is of 1 calendar; there is no history of all calendars ever
        applied.
        '''
        global staff
        try:
            if my_cal.been_applied is True:
                staff.apply_calendar(my_cal, undo=True)
                my_cal.been_applied = False
                print("Last generated calendar unapplied successfully")
                self.save()
            else:
                raise ValueError(
                    "Can only undo the last generated calendar."
                    "The last generated calendar has not been applied.")
        except NameError:
            e = NameError('No applied calendar found')
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))

    def do_reset(self, arg):
        '''
        Usage: reset <confirmation>
        
        Sets all points to 0. Cannot be undone.
        
        Parameters
        ----------
        confirmation : str
            user confirmation of 'yes' required to reset all points
        '''
        if arg == 'yes':
            for RA in staff.RAs:
                RA.points = 0
            try:
                my_cal.been_applied = False
            except:
                pass
            self.save()
        else:
            eprint("User confirmation required")

    def do_save(self, arg):
        '''
        Usage: save <filename>
        
        Saves the last generated calendar as a .csv to the 
        folder 'saved_calendars'. Open with excel or other similar app.
        
        Parameters
        ----------
        filename : str
            name of the calendar to be saved
        '''
        if arg == '':
            self.do_help("save")
            return

        if not re.match(r"^[\w\-.][\w\-. ]*$", arg):
            eprint("Invalid filename")
            return
        if arg[-4:].lower() != '.csv':
            arg += ".csv"
        path = os.path.join(saved_path, arg)
        try:
            my_cal.save_calendar_representation_csv(path)
            with open(last_cal_path, 'wb') as f:
                pickle.dump(my_cal, f)
            print("Calendar saved to "
                  f"{Fore.LIGHTBLUE_EX}{path}{Style.RESET_ALL}")
        except NameError:
            e = NameError("You have not generated any calendars")
            eprint(repr(e))
        except Exception as e:
            eprint(repr(e))

    def do_exit(self, arg):
        '''
        Exits the calgen app
        '''
        return True

    # handles most reasonable argument formats when 
    # arrays of int are involved
    @staticmethod
    def parse_int(string):
        days = []
        try:
            args = re.split(r'[,\s]+', string.strip())
            if args == ['']:
                return []
            for a in args:
                try:
                    days.append(int(a))
                except:
                    start, end = re.split("\s*:\s*", a.strip())
                    [days.append(i) for i in range(int(start),int(end) + 1)]
        except:
            raise
        return np.unique(days)

    def save(self):
        try:
            with open(staff_path, 'wb') as f:
                pickle.dump(staff, f)
        except Exception as e:
            eprint(repr(e))
            eprint("Unable to save. If you see this, something is very" 
                  " wrong in the program files. Reinstall recommended."
                  " Changes will not be reflected on next launch.")
        try:
            my_cal.recalculate(staff)
            with open(last_cal_path, 'wb') as f:
                pickle.dump(my_cal, f)
        except:
            pass
        
    # * overwritten methods from parent class

    # do nothing if user enters nothing. Done to prevent user from 
    # accidentally running a command multiple times, which is the default
    def emptyline(self):
        return

    # say goodbye
    def postloop(self):
        # close colorama
        deinit()
        # Some motivational quotes, some funny quotes, some random quotes
        chosen = np.random.choice(quotes)
        print("")
        print(chosen)
    

def main():
    shell = StaffCmd()
    shell.cmdloop()


if __name__ == "__main__":
    main()
