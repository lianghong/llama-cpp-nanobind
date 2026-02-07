# 小型初创公司Arcee AI从零开始构建了一个400B参数的开源大模型，以超越Meta的Llama | TechCrunch

> 来源：<https://techcrunch.com/2026/01/28/tiny-startup-arcee-ai-built-a-400b-open-source-llm-from-scratch-to-best-metas-llama/>

> 一家仅有30人的初创公司Arcee AI发布了名为Trinity的400B模型，该公司称这是美国公司发布的最大规模之一的开源基础模型。

---

业内大多数人 [think the winners of the AI model market](https://techcrunch.com/2025/11/03/elad-gil-on-which-ai-markets-have-winners-and-which-are-still-wide-open/) 已经达成共识：大型科技公司将主导这一领域（谷歌、Meta、微软，以及少量亚马逊），并由其选择的模型开发商——主要是OpenAI和Anthropic。

但这家仅有30人的小型初创公司 [Arcee AI](https://www.arcee.ai/) 不认同这一观点。该公司刚刚发布了一款真正且永久开放（Apache许可）的通用基础模型Trinity，Arcee声称该模型拥有400B参数，在美国公司所训练并发布的开源基础模型中规模名列前茅。

根据使用基线模型（极少进行后期训练）开展的基准测试，Arcee表示Trinity与Meta的Llama 4 Maverick 400B以及中国清华大学推出的Z.ai GLM-4.5——一款高性能开源模型——相当。

![Arcee AI benchmarks for Trinity LLM](https://techcrunch.com/wp-content/uploads/2026/01/Arcee-Benchmarks-trinity-large-preview-base.png?w=680)

Arcee AI为Trinity大语言模型（预览版，基线模型）进行的基准测试**图片来源：** Arcee AI

与其它先进水平（SOTA）模型类似，Trinity专为编程和多步骤流程（如代理系统）设计。不过尽管规模庞大，目前它仍不是真正的SOTA竞争者，因为当前仅支持文本。

更多功能正在开发中——视觉模型目前正在研发，语音转文字版本也已列入路线图。CTO Lucas Atkins向TechCrunch透露（如上图左侧所示）。相比之下，Meta的Llama 4 Maverick已是多模态模型，支持文本与图像。

但在增加更多AI模式之前，Arcee表示他们希望先打造一款能令其主要目标客户——开发者和学术界人士印象深刻的基础大模型。该公司尤其希望吸引美国各类规模的企业，摆脱选择中国开源模型的倾向。

“最终赢得这场竞赛并真正获得用户青睐的方法，是拥有最佳的开放权重模型，”Atkins表示，“要赢得开发者的认可与支持，就必须提供最好的工具。”

Techcrunch活动

### TechCrunch创始人峰会2026：门票现已开放

#### 于**波士顿，6月23日**举行，超过**1,100位创始人**齐聚一堂，在全天聚焦增长、执行和真实世界扩展的TechCrunch创始人峰会2026上，向塑造行业的创始者与投资者学习。结识处于相似成长阶段的同行伙伴，并立即带走可付诸实践的战略方法。单人票最高节省**300美元**，或通过四人及以上团队购票享受最多**30%折扣**。

### TechCrunch创始人峰会：门票现已开放

#### 于**波士顿，6月23日**举行，超过**1,100位创始人**齐聚一堂，在全天聚焦增长、执行和真实世界扩展的TechCrunch创始人峰会2026上，向塑造行业的创始者与投资者学习。结识处于相似成长阶段的同行伙伴，并立即带走可付诸实践的战略方法。单人票最高节省**300美元**，或通过四人及以上团队购票享受最多**30%折扣**。

波士顿，马萨诸塞州  
|  
6月23日，2026年

[REGISTER NOW](https://techcrunch.com/events/techcrunch-founder-summit-2026/?utm_source=tc&utm_medium=ad&utm_campaign=tcfoundersummit2026&utm_content=seb&promo=tc_inline_seb&display=)

基准测试显示，在目前仍处于预览阶段、正在进行更多后期训练的情况下，Trinity基线模型总体表现良好，部分任务甚至略胜Llama一筹——包括编程与数学、常识、知识和推理能力。

Arcee迄今所取得的进步令人印象深刻。大型Trinity模型紧随 [two previous small models](https://www.arcee.ai/blog/the-trinity-manifesto) 于12月发布的成果：一个拥有26B参数的Trinity Mini，是一款完全经过后期训练、适用于从网页应用到代理任务等各类场景的推理模型；以及一款6B参数的Trinity Nano——一个实验性模型，旨在探索极小但又高度互动模型的边界。

关键在于，Arcee仅用2000万美元，在6个月内使用了2,048块Nvidia Blackwell B300 GPU完成了所有模型的训练。创始人兼CEO Mark McQuade（如上图右侧所示）表示，这仅占公司至今融资总额约5000万美元的一小部分。

“对我们来说这笔钱可不少，”负责模型构建工作的Atkins表示。但他也承认，与目前其他大型实验室的投入相比仍相形见绌。

“六个月的时间线是经过精心计算的，”Atkins说。他此前从事汽车语音代理系统的开发工作。“我们是一家年轻且极具野心的初创公司，拥有大量人才和优秀的年轻研究人员——当我们给予他们如此多的资金并训练如此大规模模型的机会时，我深信他们会迎接挑战。而事实证明他们做到了，经历了无数个不眠之夜和长时间的奋战。”

McQuade曾是开源模型平台Hugging Face的早期员工，他表示Arcee最初并非想成为一家新的美国AI实验室：公司原本专注于为大型企业客户（如SK电信）提供模型定制服务。

“我们当时只做后期训练。我们会采用他人的优秀成果——比如Llama、Mistral或Qwen等开源模型，然后对其进行后处理以更好地满足客户的使用需求，”他表示，“包括强化学习。”

但随着客户数量的增长，Atkins表示对自主模型的需求已变得不可或缺，McQuade也开始担心依赖其他公司。与此同时，许多最先进的开源模型来自中国，而美国企业对此类来源普遍持谨慎态度，甚至被禁止使用。

这是一个令人紧张的决定。“我认为全球范围内只有不到20家公司曾达到Arcee目前所追求的规模和水平，并成功预训练并发布过自己的模型，”McQuade说。

公司最初从小项目起步，与培训公司DatologyAI合作开发了一个仅4.5B参数的小型模型。该项目的成功激励了他们开展更大规模的尝试。

但如果美国已有Llama，为何还需要另一个开源权重模型？Atkins表示，通过选择开源Apache许可协议，该公司承诺始终将模型保持开放。此举发生在Meta首席执行官马克·扎克伯格去年 [indicated his company might not always](https://techcrunch.com/2025/07/30/zuckerberg-says-meta-likely-wont-open-source-all-of-its-superintelligence-ai-models/) 将其所有最先进模型全部开源之后。

“Llama不能被视为真正意义上的开源，因为它采用Meta控制的许可证，并附带商业使用和用途限制。”他说，“这导致 [some open source organizations to claim](https://opensource.org/blog/metas-llama-license-is-still-not-open-source) 认为Llama根本不符合开源合规标准。”

“Arcee的存在，是因为美国需要一个永久开放、采用Apache许可协议且真正能与当今前沿水平竞争的替代方案，”McQuade表示。

所有Trinity模型——无论大小均可免费下载。最大版本将发布三种形态：Trinity Large Preview是经过轻度后期训练的指令型模型，意味着它已被训练以遵循人类指令，而不仅仅是预测下一个词，因此适用于通用对话场景；Trinity Large Base则是未经后期训练的基线模型。

接着是TrueBase，该模型不包含任何指令数据或后续训练内容，使企业或研究人员在定制时无需处理原始数据、规则或假设。

Arcee AI最终将提供其通用发布模型的托管版本，并宣称将以具有竞争力的API价格推出。该产品预计在六周内上线，初创公司仍将持续优化模型的推理训练。

Trinity Mini的API定价为每千次请求0.045美元/0.15美元，同时提供有限速率的免费版本。此外，公司仍在销售后期训练与定制服务选项。

主题

[AI](https://techcrunch.com/category/artificial-intelligence/), [Arcee AI](https://techcrunch.com/tag/arcee-ai/), [foundation models](https://techcrunch.com/tag/foundation-models/), [llama 4](https://techcrunch.com/tag/llama-4/), [open source ai](https://techcrunch.com/tag/open-source-ai/), [Startups](https://techcrunch.com/category/startups/), [Trinity](https://techcrunch.com/tag/trinity/)

![Julie Bort](https://techcrunch.com/wp-content/uploads/2025/08/julie-bort-disrupt.jpg?w=150)

朱莉·博特

风险投资编辑

朱莉·博特是TechCrunch的初创企业/风投版块主编。

您可以通过发送邮件至 [julie.bort@techcrunch.com](mailto:julie.bort@techcrunch.com) 或通过X平台上的 [@Julie188](https://x.com/Julie188) 与她联系或核实联络信息。

[View Bio](https://techcrunch.com/author/julie-bort/)

![Event Logo](https://techcrunch.com/wp-content/uploads/2025/07/TC25_Disrupt-Color.png)

10月13日至15日

旧金山，加利福尼亚州

**票务现已开放，价格为全年最低。** 购买通行证可立省高达 680 美元——若你是 **前 500 名注册者**，还可额外获得一张 **半价通行证**。

与投资者见面。发现你下一个投资组合公司。聆听来自 **250 多位科技领袖**的分享，参与 **200 多场讲座**，探索 **300 多家正在打造未来的新创企业**。别错过这些限时优惠。

[**REGISTER NOW**](https://techcrunch.com/events/tc-disrupt-2026/?utm_source=tc&utm_medium=ad&utm_campaign=disrupt2026&utm_content=sebbogo&promo=rightrail_sebbogo&display=)

## 最受欢迎

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