# Tiny startup Arcee AI built a 400B-parameter open source LLM from scratch to best Meta's Llama | TechCrunch

> Source: <https://techcrunch.com/2026/01/28/tiny-startup-arcee-ai-built-a-400b-open-source-llm-from-scratch-to-best-metas-llama/>

> 30-person startup Arcee AI has released a 400B model called Trinity, which it says is one of the biggest open source foundation models from a U.S. company.

---

Many in the industry [think the winners of the AI model market](https://techcrunch.com/2025/11/03/elad-gil-on-which-ai-markets-have-winners-and-which-are-still-wide-open/) have already been decided: Big Tech will own it (Google, Meta, Microsoft, a bit of Amazon) along with their model makers of choice, largely OpenAI and Anthropic.

But tiny 30-person startup [Arcee AI](https://www.arcee.ai/) disagrees. The company just released a truly and permanently open (Apache license) general-purpose, foundation model called Trinity, and Arcee claims that at 400B parameters, it is among the largest open source foundation models ever trained and released by a U.S. company.

Arcee says Trinity compares to Meta’s Llama 4 Maverick 400B, and Z.ai’s GLM-4.5, a high-performing open source model from China’s Tsinghua University, according to benchmark tests conducted using base models (very little post-training).

![Arcee AI benchmarks for Trinity LLM](https://techcrunch.com/wp-content/uploads/2026/01/Arcee-Benchmarks-trinity-large-preview-base.png?w=680)

Arcee AI benchmarks for its Trinity large LLM (preview version, base model)**Image Credits:**Arcee AI

Like other state-of-the-art (SOTA) models, Trinity is geared for coding and multi-step processes like agents. Still, despite its size, it’s not a true SOTA competitor yet because it currently supports only text.

More modes are in the works — a vision model is currently in development, and a speech-to-text version is on the roadmap, CTO Lucas Atkins told TechCrunch (pictured above, on the left). In comparison, Meta’s Llama 4 Maverick is already multi-modal, supporting text and images.

But before adding more AI modes to its roster, Arcee says, it wanted a base LLM that would impress its main target customers: developers and academics. The team particularly wants to woo U.S. companies of all sizes away from choosing open models from China.

“Ultimately, the winners of this game, and the only way to really win over the usage, is to have the best open-weight model,” Atkins said. “To win the hearts and minds of developers, you have to give them the best.”

Techcrunch event

### TechCrunch Founder Summit 2026: Tickets Live

#### On **June 23 in Boston**, more than **1,100 founders** come together at **TechCrunch Founder Summit 2026** for a full day focused on growth, execution, and real-world scaling. Learn from founders and investors who have shaped the industry. Connect with peers navigating similar growth stages. Walk away with tactics you can apply immediately Save **up to $300** on your pass or **save up to 30% with group tickets for teams of four or more.**

### TechCrunch Founder Summit: Tickets Live

#### On **June 23 in Boston**, more than **1,100 founders** come together at **TechCrunch Founder Summit 2026** for a full day focused on growth, execution, and real-world scaling. Learn from founders and investors who have shaped the industry. Connect with peers navigating similar growth stages. Walk away with tactics you can apply immediately Save **up to $300** on your pass or **save up to 30% with group tickets for teams of four or more.**

Boston, MA
|
June 23, 2026

[REGISTER NOW](https://techcrunch.com/events/techcrunch-founder-summit-2026/?utm_source=tc&utm_medium=ad&utm_campaign=tcfoundersummit2026&utm_content=seb&promo=tc_inline_seb&display=)

The benchmarks show that the Trinity base model, currently in preview while more post-training takes place, is largely holding its own and, in some cases, slightly besting Llama on tests of coding and math, common sense, knowledge, and reasoning.

The progress Arcee has made so far to become a competitive AI Lab is impressive. The large Trinity model follows [two previous small models](https://www.arcee.ai/blog/the-trinity-manifesto) released in December: the 26B-parameter Trinity Mini, a fully post-trained reasoning model for tasks ranging from web apps to agents, and the 6B-parameter Trinity Nano, an experimental model designed to push the boundaries of models that are tiny yet chatty.

The kicker is, Arcee trained them all in six months for $20 million total, using 2,048 Nvidia Blackwell B300 GPUs. This out of the roughly $50 million the company has raised so far, said founder and CEO Mark McQuade (pictured above, on the right).

That kind of cash was “a lot for us,” said Atkins, who led the model-building effort. Still, he acknowledged that it pales in comparison to how much bigger labs are spending right now.

The six-month timeline “was very calculated,” said Atkins, whose career before LLMs involved building voice agents for cars. “We are a younger startup that’s extremely hungry. We have a tremendous amount of talent and bright young researchers who, when given the opportunity to spend this amount of money and train a model of this size, we trusted that they’d rise to the occasion. And they certainly did, with many sleepless nights, many long hours.”

McQuade, previously an early employee at open source model marketplace Hugging Face, says Arcee didn’t start out wanting to become a new U.S. AI lab: The company was originally doing model customization for large enterprise clients like SK Telecom.

“We were only doing post-training. So we would take the great work of others: We would take a Llama model, we would take a Mistral model, we would take a Qwen model that was open source, and we would post-train it to make it better” for a company’s intended use, he said, including doing the reinforcement learning.

But as their client list grew, Atkins said, the need for their own model was becoming a necessity, and McQuade was worried about relying on other companies. At the same time, many of the best open models were coming from China, which U.S. enterprises were leery of, or were barred from using.

It was a nerve-wracking decision. “I think there’s less than 20 companies in the world that have ever pre-trained and released their own model” at the size and level that Arcee was gunning for, McQuade said.

The company started small at first, trying its hand at a tiny, 4.5B model created in partnership with training company DatologyAI. The project’s success then encouraged bigger endeavors.

But if the U.S. already has Llama, why does it need another open weight model? Atkins says by choosing the open source Apache license, the startup is committed to always keeping its models open. This comes after Meta CEO Mark Zuckerberg last year [indicated his company might not always](https://techcrunch.com/2025/07/30/zuckerberg-says-meta-likely-wont-open-source-all-of-its-superintelligence-ai-models/) make all of its most advanced models open source.

“Llama can be looked at as not truly open source as it uses a Meta-controlled license with commercial and usage caveats,” he says. This has caused [some open source organizations to claim](https://opensource.org/blog/metas-llama-license-is-still-not-open-source) that Llama isn’t open source compliant at all.

“Arcee exists because the U.S. needs a permanently open, Apache-licensed, frontier-grade alternative that can actually compete at today’s frontier,” McQuade said.

All Trinity models, large and small, can be downloaded for free. The largest version will be released in three flavors. Trinity Large Preview is a lightly post-trained instruct model, meaning it’s been trained to follow human instructions, not just predict the next word, which gears it for general chat usage. Trinity Large Base is the base model without post-training.

Then we have TrueBase, a model with any instruct data or post training so enterprises or researchers that want to customize it won’t have to unroll any data, rules, or assumptions.

Arcee AI will eventually offer a hosted version of its general-release model for, it says, competitive API pricing. That release is up to six weeks away as the startup continues to improve the model’s reasoning training.

API pricing for Trinity Mini is $0.045 / $0.15, and there is a rate-limited free tier available, too. Meanwhile, the company still sells post-training and customization options.

Topics

[AI](https://techcrunch.com/category/artificial-intelligence/), [Arcee AI](https://techcrunch.com/tag/arcee-ai/), [foundation models](https://techcrunch.com/tag/foundation-models/), [llama 4](https://techcrunch.com/tag/llama-4/), [open source ai](https://techcrunch.com/tag/open-source-ai/), [Startups](https://techcrunch.com/category/startups/), [Trinity](https://techcrunch.com/tag/trinity/)

![Julie Bort](https://techcrunch.com/wp-content/uploads/2025/08/julie-bort-disrupt.jpg?w=150)

Julie Bort

Venture Editor

Julie Bort is the Startups/Venture Desk editor for TechCrunch.


You can contact or verify outreach from Julie by emailing [julie.bort@techcrunch.com](mailto:julie.bort@techcrunch.com) or via [@Julie188](https://x.com/Julie188) on X.

[View Bio](https://techcrunch.com/author/julie-bort/)

![Event Logo](https://techcrunch.com/wp-content/uploads/2025/07/TC25_Disrupt-Color.png)

October 13-15

San Francisco, CA

**Tickets are live at the lowest rates of the year.** Save up to $680 on your pass — and if you’re among the **first 500 registrants,** score a **+1 pass at 50% off**.

Meet investors. Discover your next portfolio company. Hear from **250+ tech leaders**, dive into **200+ sessions**, and explore **300+ startups** building what’s next. Don’t miss these one-time savings.

[**REGISTER NOW**](https://techcrunch.com/events/tc-disrupt-2026/?utm_source=tc&utm_medium=ad&utm_campaign=disrupt2026&utm_content=sebbogo&promo=rightrail_sebbogo&display=)

## Most Popular

- ### [Tesla is killing off the Model S and Model X](https://techcrunch.com/2026/01/28/tesla-is-killing-off-the-model-s-and-model-x/)

  - [Sean O'Kane](https://techcrunch.com/author/sean-okane/)
- ### [Meta to test premium subscriptions on Instagram, Facebook, and WhatsApp](https://techcrunch.com/2026/01/26/meta-to-test-premium-subscriptions-on-instagram-facebook-and-whatsapp/)

  - [Aisha Malik](https://techcrunch.com/author/aisha-malik/)
- ### [Anthropic launches interactive Claude apps, including Slack and other workplace tools](https://techcrunch.com/2026/01/26/anthropic-launches-interactive-claude-apps-including-slack-and-other-workplace-tools/)

  - [Russell Brandom](https://techcrunch.com/author/russell-brandom/)
- ### [This founder cracked firefighting — now he’s creating an AI gold mine](https://techcrunch.com/2026/01/25/this-founder-cracked-firefighting-now-hes-creating-an-ai-gold-mine/)

  - [Connie Loizos](https://techcrunch.com/author/connie-loizos/)
- ### [TikTok users freak out over app’s ‘immigration status’ collection — here’s what it means](https://techcrunch.com/2026/01/23/tiktok-users-freak-out-over-apps-immigration-status-collection-heres-what-it-means/)

  - [Sarah Perez](https://techcrunch.com/author/sarah-perez/)
- ### [Researchers say Russian government hackers were behind attempted Poland power outage](https://techcrunch.com/2026/01/23/researchers-say-russian-government-hackers-were-behind-attempted-poland-power-outage/)

  - [Zack Whittaker](https://techcrunch.com/author/zack-whittaker/)
- ### [Microsoft gave FBI a set of BitLocker encryption keys to unlock suspects’ laptops: Reports](https://techcrunch.com/2026/01/23/microsoft-gave-fbi-a-set-of-bitlocker-encryption-keys-to-unlock-suspects-laptops-reports/)

  - [Lorenzo Franceschi-Bicchierai](https://techcrunch.com/author/lorenzo-franceschi-bicchierai/)
