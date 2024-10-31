---
layout: distill
title: Bonding 5/21 - 5/27
description: Analysis of code replication and literature review
tags: Weekly_blogs
categories: gsoc-2024
date: 2024-05-26
permalink: /blog/2024/community-bonding-5-21-5-27/
# featured: true

authors:
  - name: Zebin Huang
    affiliations:
      name: Edinburgh Centre for Robotics

bibliography: bonding_week.bib
---


### Weekly Meeting

During the May 20, 2024, we discussed beginning the project by replicating last year's model using a simple LLM for handling different input commands and gradually progressing towards more complex models. Key tasks for moving forward include conducting a literature review to define the project's specific research question, setting up necessary tools like CARLA and behavior metrics, and addressing technical setup challenges.

More details can be found here: [Google Doc](https://docs.google.com/document/d/1b2ZEU5Gt8gP2ae_YzNSJSd7RukUrsG_aDJFLnbvoQiM/edit)

### To-Do List during the bonding period

- [x]  Set up a blog based on examples from previous years.
- [x]  Set up CARLA.
- [x]  Run Qi's models.
- [x]  Read and analyze literature on autonomous driving and LLMs.

### **Code Replication**

This week, I attempted to replicate certain elements of the project codebase [Meiqizhao's code](https://github.com/TheRoboticsClub/gsoc2023-Meiqi_Zhao) and encountered some challenges that required raising issues for resolution. Specifically, I opened two issues [issue1](https://github.com/TheRoboticsClub/gsoc2023-Meiqi_Zhao/issues/2) [issue2](https://github.com/TheRoboticsClub/gsoc2023-Meiqi_Zhao/issues/3) and [one PR](https://github.com/TheRoboticsClub/gsoc2023-Meiqi_Zhao/pull/4) regarding bugs found during the replication process. To enhance reproducibility, I am currently working with Docker, and I also plan to provide a Docker branch later on.

### **Behavior Metrics Exploration**

I reviewed the [Behaviour Metrics](https://github.com/JdeRobot/BehaviorMetrics) repos and related papers. The Behavior Metrics can provide a structured framework for quantifying the effectiveness and  performance of autonomous system in simulated scenarios. Incorporating text input for autonomous driving guidance enhances the Behavior Metrics benchmark by interactivity and interpretability. Here are some potential integration methods and benefits:

1. **Expanded Testing Scenarios**: It enables the creation of a broader range of test environments and situations that include verbal commands and interactions.
2. **Enhanced Textual Interpretability**: Provides clarity on how the system interprets and responds to natural language inputs, which improves the system's transparency and trustworthiness.
3. **Adapted Interaction Methods**: Allows for modifications in user interaction, offering more intuitive and accessible ways for users to communicate with autonomous systems.

### **Literature Review and Feasibility Analysis**

I conducted a review on research papers related to our project. The focus was on assessing the feasibility of replicating the studies, considering factors like data availability, computational requirements, and whether the methods are open-source. Such analysis helps in understanding the practical aspects of implementing these research findings in our work.

| Paper Title                                                                                     | Reproducibility                                                                                                  | Data Volume                                                                     | Technical Difficulty                                                                             | GPU Requirements                                                                            |
| ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| GPT-4V Takes the Wheel <d-cite key="huangGPT4VTakesWheel2024a"></d-cite>                        | Low: Uses publicly available datasets. <span style="color:red;">Not open-sourced</span>                          | JAAD, WiDEVIEW                                                                  | High: Integrates vision and language models for dynamic behavior prediction                      | High: VLM processing but not illustrated                                                    |
| Driving with LLMs <d-cite key="chenDrivingLLMsFusing2023a"></d-cite>                            | Low: New dataset and unique architecture, reproducibility [GitHub](https://github.com/wayveai/Driving-with-LLMs) | Custom 160k QA pairs, 10k driving scenario. Which simulator?                    | Very High: Novel fusion of vector modalities and LLMs                                            | Moderate: Minimum of 20GB VRAM for running evaluations, Minimum of 40GB VRAM for training   |
| LMDrive <d-cite key="shaoLMDriveClosedLoopEndtoEnd2023a"></d-cite>                              | High: Dataset and models are open-sourced, but complexity in GPU setup                                           | <span style="color:green;">64K parsed clips and 464K notice instructions</span> | Very High: Real-time, closed-loop control with LLMs in vehicles                                  | <span style="color:red;">Very High: 2~3 days for the visual encoder on 8x A100 (80G)</span> |
| Language Models as Trajectory Generators <d-cite key="kwonLanguageModelsZeroShot2023"></d-cite> | High: Standard dataset, clear methodology and evaluation process                                                 | Flexible data generation with Pybullet                                          | Moderate: Focus on trajectory generation using LLMs, less complex than real-time control systems | Low: Less demanding compared to real-time visual tasks                                      |


<p></p>
Here is a summary of the preliminary analysis of different literature pieces:
1. **GPT-4V Takes the Wheel**: This work utilizes publicly available datasets but is not open-sourced, which poses a significant barrier to reproducibility. Although it can serve as a conceptual reference, the lack of open access means it cannot be directly replicated.
2. **Driving with LLMs**: The source code is open. However, the simulator used is proprietary to Wayve, restricting access and thus full replication of the project. While the architecture and approach can be studied.
3. **LMDrive**: This project appears the most promising in terms of openness and practical usability. It is conducted on the Carla simulator platform, and pre-trained models along with the dataset are provided. Although there are no current reproducibility issues or bugs reported, the main challenge is the significant computational requirement—training requires eight A100 GPUs (80GB each). Initial testing might focus on evaluating the provided pre-trained models due to these resource demands.
4. **Language Models as Trajectory Generators**: This work offers a unique perspective by using zero-shot methods in manipulators, which is the least resource-intensive approach among the ones listed. However, for real-time systems like autonomous driving, this approach would need to incorporate more robust and safer control mechanisms to ensure reliability and safety in dynamic environments.

From the feasibility standpoint, some of the literature reviewed indicated very high resource requirements, such as one paper necessitating 8 * A100 GPUs. These are substantial resource demands that pose challenges for replication.

The core question we need to address is: What is our objective? If the goal is to replicate existing solutions and integration, we need to identify the features and MVP. However, if our aim is to optimize, the biggest hurdle is the training phase, particularly the GPU bottlenecks during this process. This will need to be discussed further in next week's meeting.

### **Moving Forward**

Understanding these resource limitations and objectives will help guide our project's direction. Our next steps involve deciding whether to seek resource optimization or to focus on adapting our goals to fit the available computational resources. Additionally, we are currently addressing several issues and plan to conduct further literature research to deepen my understanding of the field.

<!-- ### References

[1] S. Paniego, R. Calvo-Palomino, and J. Cañas, ‘Behavior metrics: An open-source assessment tool for autonomous driving tasks’, *SoftwareX*, vol. 26, May 2024, doi: [10.1016/j.softx.2024.101702](https://doi.org/10.1016/j.softx.2024.101702).

[2] J. Huang, P. Jiang, A. Gautam, and S. Saripalli, ‘GPT-4V Takes the Wheel: Promises and Challenges for Pedestrian Behavior Prediction’. arXiv, Jan. 25, 2024.

[3] H. Shao, Y. Hu, L. Wang, S. L. Waslander, Y. Liu, and H. Li, ‘LMDrive: Closed-Loop End-to-End Driving with Large Language Models’. arXiv, Dec. 21, 2023. doi: [10.48550/arXiv.2312.07488](https://doi.org/10.48550/arXiv.2312.07488).

[4] L. Chen *et al.*, ‘Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving’. arXiv, Oct. 13, 2023. doi: [10.48550/arXiv.2310.01957](https://doi.org/10.48550/arXiv.2310.01957).

[5] T. Kwon, N. Di Palo, and E. Johns, ‘Language Models as Zero-Shot Trajectory Generators’. arXiv, Oct. 17, 2023. doi: [10.48550/arXiv.2310.11604](https://doi.org/10.48550/arXiv.2310.11604). -->
