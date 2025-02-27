# Memory Box System Prompts

This directory contains system prompts used with the Memory Box plugin. These prompts guide AI assistants in effectively storing and retrieving memories using structured formats.

## How the System Prompt Works

The system prompt (`system.md`) is designed to enhance the AI's ability to create well-structured, retrievable memories through the Memory Box plugin. It works by:

1. **Providing Clear Structure**: Defines specific formats for different types of information (technical details, decisions, solutions, etc.), ensuring consistency across memories.

2. **Optimizing for Retrieval**: Encourages the inclusion of searchable keywords and consideration of how users might search for the information in the future.

3. **Balancing Technical and User-Friendly Language**: Instructs the AI to include both technical terms and user-friendly alternatives, making memories accessible to users with varying levels of expertise.

4. **Enforcing Conciseness**: Sets guidelines for memory length (50-150 words) to ensure memories contain sufficient context without unnecessary details.

5. **Guiding Memory Testing**: Prompts the AI to evaluate memories before storage by considering potential search terms.

## Prompt Components Explained

- **Focused Memories**: Each memory should contain a single concept or topic to avoid confusion and improve retrievability.

- **Structured Formats**: Different information types (TECHNICAL, DECISION, SOLUTION, etc.) have specific formats to ensure consistent organization.

- **Diverse Terminology**: Including both technical and user-friendly terms improves the chances of matching a user's search query.

- **Searchable Keywords**: Starting with common search terms increases the likelihood of retrieval.

- **Detail Balance**: Including both high-level descriptions and technical details makes memories useful for different purposes.

## Prompt Engineering and AI Integration

Effective prompt engineering is crucial for maximizing the value of AI tools like Memory Box. A well-crafted system prompt can significantly improve:

- **Information Organization**: Structured prompts lead to better-organized information.
- **Retrieval Accuracy**: Properly formatted memories are easier to find when needed.
- **User Experience**: Clear guidelines result in more consistent and useful AI responses.
- **Knowledge Management**: Systematic approaches to memory storage create more valuable knowledge bases over time.

## About Amotivv, Inc.

Amotivv, Inc. specializes in developing AI-powered solutions that enhance productivity and knowledge management. Our expertise includes:

- **Custom AI Integration**: Tailoring AI solutions to specific business needs and workflows.
- **Prompt Engineering**: Designing effective prompts that maximize the value of large language models.
- **Knowledge Management Systems**: Building systems like Memory Box that help organizations capture and retrieve institutional knowledge.
- **AI Training and Consultation**: Helping teams effectively leverage AI tools through training and best practices.

Amotivv can help your organization implement similar memory solutions, enhancing your team's ability to store, retrieve, and leverage knowledge. Our approach combines technical expertise with a deep understanding of how users interact with AI systems.

For more information about how Amotivv can help with your AI integration needs, contact us at [contact@amotivv.com](mailto:contact@amotivv.com).

## Version Control

This repository uses Git for version control of system prompts. When making changes to the system prompt:

1. Update the `system.md` file directly
2. Use descriptive commit messages explaining the changes and their purpose
3. Use `git log -p prompts/system.md` to view the history of changes

This approach allows for tracking the evolution of prompts over time while maintaining a single source of truth.
