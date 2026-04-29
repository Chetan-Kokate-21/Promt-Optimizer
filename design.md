# Prompt Optimizer UI Design Brief

## Product
Design a modern web interface for a tool called **Prompt Optimizer**.

The product accepts a user prompt, runs it through a multi-stage optimization pipeline, and returns:
- the final optimized prompt
- evaluation metrics
- pipeline visualization
- domain/context expansion details

This interface should feel polished, academic-demo ready, and appropriate for showing to a professor or project reviewer.

## Primary Goal
Create a UI that makes the system feel:
- technically credible
- easy to understand
- visually elegant
- presentation-friendly

The interface should communicate that this is not just a text box and result box. It is a system with:
- rule-based optimization
- ML-guided optimization
- genetic algorithm optimization
- domain-aware context expansion

## Audience
- students presenting a project
- professors evaluating technical depth
- developers testing prompt optimization behavior

## Key UX Goals
- Make the optimization flow easy to follow visually.
- Make the final optimized prompt the hero output.
- Clearly show how the system works without overwhelming the user.
- Present technical information in a structured and visually refined way.
- Support both desktop and mobile layouts.

## Core Screens
Design a single-page app with these sections:

### 1. Hero / Header
Include:
- Product name: `Prompt Optimizer`
- Subtitle explaining that it uses rule-based, ML-guided, and genetic optimization
- A short supporting sentence about improving prompt quality, context, and structure

Tone:
- polished
- confident
- technical but accessible

### 2. Prompt Input Panel
Include:
- large prompt textarea
- mode selector with:
  - `Cost`
  - `Context`
- primary CTA button: `Optimize Prompt`
- secondary button: `Load Demo`

Optional helper text:
- Cost mode focuses on shorter, efficient prompts
- Context mode focuses on richer, domain-aware prompts

### 3. Optimized Prompt Panel
This should be the most visually important output region.

Include:
- section title: `Optimized Prompt`
- large formatted display area
- copy button
- optional export/download button

The optimized prompt should be displayed with structured section styling:
- Role
- Task
- Context
- Constraints
- Output
- Example

The display should look like a premium prompt document rather than plain JSON output.

### 4. Metrics Panel
Include cards for:
- Token Count Before
- Token Count After
- Semantic Score
- Improvement Score

Style:
- clean metric cards
- visually scannable
- subtle emphasis on improvement

### 5. Pipeline Visualization Panel
This is very important.

Show 3 stages:
1. Rule-Based Rewrite
2. ML-Guided Rewrite
3. Genetic Algorithm

For each stage show:
- whether it ran
- whether it was selected as final
- stage description
- key metrics

If possible, use a vertical or horizontal process diagram with clear states:
- ran
- selected
- compared

If the genetic stage has history, show generation cards or timeline entries such as:
- generation number
- best fitness
- winning strategy lineage

This should feel like a lightweight workflow explorer.

### 6. Domain Context Expansion Panel
Show:
- detected domain
- seed terms extracted from the prompt
- related technical terms added
- source used for expansion

Important:
This panel should help explain how the system enriches prompts with domain-specific context.

### 7. Raw Response Panel
Include a collapsible or secondary panel for raw JSON/API response.
This should not compete visually with the optimized prompt.

## Visual Direction
The current UI should be replaced with something more intentional and premium.

Desired style:
- editorial + technical
- warm but not old-fashioned
- refined, structured, high-clarity

Avoid:
- generic dashboard look
- dark cyberpunk style
- purple gradients
- default SaaS layout

Preferred design language:
- soft warm neutrals or muted sand/stone backgrounds
- deep green, slate, rust, or ink accents
- layered cards with subtle shadows
- strong typography hierarchy
- elegant spacing
- visible section rhythm

## Typography
Use expressive typography.

Suggested direction:
- Headings: a strong serif or editorial display face
- Body/UI text: a clean readable sans or refined serif/sans pairing

The page should feel designed, not templated.

## Layout
Desktop:
- two-column top workspace
- left: input area
- right: optimized output
- lower stacked panels for metrics, pipeline visualization, domain expansion, raw response

Mobile:
- single-column flow
- optimized prompt should appear early
- cards should stack cleanly

## Interaction Notes
- Show loading state while optimization runs
- Use subtle animated transitions when results appear
- Allow smooth scrolling to results after submission
- Use clear selected/running/completed visual states in the pipeline

## Accessibility
- strong text contrast
- large readable typography
- keyboard-friendly controls
- clear labels and section headers

## Content Priorities
The order of importance on the page should be:
1. Prompt input
2. Final optimized prompt
3. Pipeline visualization
4. Metrics
5. Domain context expansion
6. Raw response

## Data Model Available In UI
The UI can expect data such as:
- `optimized_prompt`
- `metrics`
- `details.brain_decision.chosen_stage`
- `details.brain_decision.brain_trace.stages.rule_based`
- `details.brain_decision.brain_trace.stages.ml_guided`
- `details.brain_decision.brain_trace.stages.genetic`
- `details.domain_term_expansion.domain`
- `details.domain_term_expansion.seed_terms`
- `details.domain_term_expansion.related_terms`
- `details.domain_term_expansion.term_source`

## Desired Deliverable From Stitch AI
Generate:
- full-page UI design
- desktop and mobile responsive layout
- polished visual hierarchy
- clear cards/sections for prompt engineering workflow
- component ideas for pipeline state visualization
- visually strong formatted prompt display

## Design Prompt For Stitch AI
Design a premium single-page web app called "Prompt Optimizer" for a technical academic project demo. The app allows users to enter a prompt, choose Cost or Context mode, and run a multi-stage optimization pipeline including rule-based rewriting, ML-guided optimization, a genetic algorithm, and domain-aware context expansion. The UI should make the final optimized prompt the hero output while also clearly visualizing the three-stage pipeline, key evaluation metrics, and domain term expansion details. Use an editorial-technical design language with strong typography, warm refined neutrals, subtle green/slate/rust accents, elegant card layouts, and clear information hierarchy. Avoid generic SaaS dashboards, dark cyberpunk themes, and purple-heavy styling. The page should be responsive, presentation-ready, and suitable for explaining the system to a professor.
