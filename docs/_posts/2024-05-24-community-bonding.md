---
layout: distill
title: Bonding 5/15 - 5/21
description: Initial setup for GSoC 2024 and summary of first meetings
tags: GSoC, Weekly_blogs, JdeRobot
categories: gsoc-2024
date: 2024-05-24
permalink: /blog/2024/community-bonding-5-15-5-21/
# featured: true

authors:
  - name: Zebin Huang
    affiliations:
      name: Edinburgh Centre for Robotics

bibliography: posts.bib
---

### GSoC Setting Things

Completed initial setup tasks for GSoC 2024, including payment registration, joining the Discord channel, and participating in the May 7th Contributor Summit. The summit was a great opportunity to connect with developers worldwide and gain valuable insights and tips for the program. Engaged with my mentor and community, reviewed roles and responsibilities, updated my display name, and logged meetings for future reference.

### Meeting with Mentors (5/9)

During our first meeting, I had the opportunity to get acquainted with my mentors and discuss the project's initial steps. Apoorv shared his journey from being a GSoC contributor to becoming a mentor at JdeRobot. David, a first-year mentor at GSoC and a PhD student, provided insights into his research on vision-based detection and drone localization.

We discussed the technical setup for the project, including GPU access. Communication was emphasized, with Slack being the primary tool for ongoing discussions. Apoorv explained the usage of our current repository, which mainly stores documents and blogs.

The discussion also covered the project's initial steps, such as reviewing relevant research papers and developing a minimum viable product. Apoorv provided instructions for setting up the Carla simulator, which will be essential for our project.

Sergio suggested starting with a simple BERT model that classifies instructions into commands, using Qi's development from last year's project as a starting point. We also discussed the possibility of publishing our results in a scientific paper after the summer, which added an exciting goal to our project.

More details can be found here: [Google Doc](https://docs.google.com/document/d/1b2ZEU5Gt8gP2ae_YzNSJSd7RukUrsG_aDJFLnbvoQiM/edit)

This document is intended to help everyone from different time zones stay on track for past meetings. It will also streamline our preparations for future meetings. I will make and review the agendas before each meeting.

#### To-Do List during the bonding period

- [x] Set up a blog based on examples from previous years.
- [x] Set up CARLA.
- [ ] Run Qi's models.
- [ ] Read and analyze literature on autonomous driving and LLMs.

### JdeRobot Kick-off Meeting (5/15)

In this meeting, we were welcomed to the JdeRobot GSoC program for 2024. A brief presentation about JdeRobot and its main projects was given, which helped in understanding the context and scope of the organization. Each participant, including myself, gave a quick introduction and described their project.

### Progress

Last week, I focused on setting up my blog and configuring the Carla simulator using Docker. I created the blog using Jekyll and [al-folio](https://github.com/alshedivat/al-folio/tree/master), referencing examples from previous years, and successfully configured the deployment workflow.

For the Carla simulator, I opted for a Docker setup, considering I am currently primarily using servers and macOS. This choice was made to simplify the initial setup process, with plans to transition to a physical machine later. The documentation available online was quite scattered, and there were differences from the official tutorials. Hence, I plan to write a separate blog later to document this setup process in detail.

Additionally, I have been reviewing [Meiqizhao's code](https://github.com/TheRoboticsClub/gsoc2023-Meiqi_Zhao) implementation and understanding the general ideas behind [behaviour metrics](https://github.com/JdeRobot/BehaviorMetrics).

Moving forward, I will continue with the setup tasks and delve deeper into the research question to ensure a solid foundation for our project.
