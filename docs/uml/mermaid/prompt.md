# Prompt given to the LLM to write the Mermaid sequence diagram

(Assignment: 25 bonus points for a UML sequence diagram in Mermaid; the LLM may be used for this syntax only, and the
prompt must be included.)

---

Write a Mermaid `sequenceDiagram` for the system described below. Use only Mermaid syntax (participant, ->>, -->>,
alt/else, loop, opt, Note, autonumber). Use the actors and the steps exactly as listed, in this order. Do not add
actors or steps of your own. Keep each message under about 12 words. Return only the Mermaid code block.

**System:** ASPIRE, an agentic system that takes a task request and produces a solution, using ARC models for every
agent except the Claude and ChatGPT calls in the re-design debate.

**Actors (participants, left to right):**
User; Interviewer (model 1); Blackboard (shared project folder); Planner 1 (model 2); Planner 2 (model 3);
Editor (model 1 or 4); Knowledge Evaluator; Librarian; Research Orchestrator; Research Agents; Wiki Verifier;
Head Orchestrator; Sub-orchestrators; Project Critic; Problem Solver 1; Final Evaluator; Auditor 1; Auditor 2;
Claude and ChatGPT APIs.

**Flow and steps:**

1. Intake. The User submits a task request. The Interviewer asks questions, clarifies the request, generates tasks and
   asks for preliminary approval of the plan (Plan A). Loop until the User is satisfied. Then a project folder is
   created on the Blackboard with a txt file defining the task.
2. Planning loop. Planner 1 reads the plan, critiques it and its tasks, edits it and writes an alternative, Plan B.
   Planner 2 reads both plans and writes pros and cons for each. The Editor edits each plan using the pros and cons.
   Planner 1, Planner 2 and the Editor all vote on which plan moves forward. The result is a refined plan with
   tasks and subtasks, written to the Blackboard.
3. Knowledge loop. The Knowledge Evaluator lists the topics and knowledge areas the task needs. The Librarian, who
   manages the wikis, reports which topics are covered. If all are covered, go to step 4. If not: the Research
   Orchestrator gives each topic or subtopic to the Research Agents, who search scholarly articles (Elsevier),
   NCBI, PMC and parts catalogs through APIs and web search, and return documents. The Research Orchestrator approves
   each document and decides whether another research round is needed (repeat the search if so). Then the Librarian
   sorts the documents into categories and starts the wiki addition loop: the Wiki Verifier checks that each
   addition is accurate (repeat until accurate). Then the Knowledge Evaluator is reactivated, checks that the
   requested information is now present, reviews the new data against the original task, and says whether more
   knowledge is needed (repeat from the research step if so). This lets a broad first request narrow to specifics.
4. Production. The Knowledge Evaluator approves. The Head Orchestrator reads project templates from the knowledge
   base, delegates tasks to the Sub-orchestrators, who delegate subtasks and return their outputs. The Head
   Orchestrator organizes the outputs together.
5. Critic loop. The Project Critic reviews the result for holes, failures, inefficiencies, ease of use and
   effectiveness, and sends the findings to Problem Solver 1 (a different model), who suggests solutions and returns
   them. The Project Critic accepts or declines. If declined: Problem Solver 1 checks the wiki for relevant topics
   and tries again; if the wiki has none, the knowledge and research loop runs again. If the solutions are declined a
   second time after the knowledge loop was reactivated, go to step 7 (re-design).
6. Final evaluation. When the Project Critic accepts, the Final Evaluator does QA: checks all files and does test
   runs. If approved, it produces a folder with the files and sends it to the User (potentially by email). If not
   approved, it returns to the Head Orchestrator with descriptions of the issues found (go back to step 4).
7. Re-design loop. Auditor 1 evaluates the successes, failures and shortcomings and what caused the poor result.
   Auditor 2 (a different model) evaluates the proposed solutions and sends suggestions to Auditor 1, who returns an
   evaluation of them. Repeat until both auditors agree on the problems and fixes. An LLM call summarizes the
   project progress, the transcript and the auditors' suggestions. The Claude and ChatGPT APIs are queried for ways to
   improve the whole system and debate each other. Stop when a fix is decided or a credit cost limit is reached.
   A report of the fix is created and sent to the User for approval.
