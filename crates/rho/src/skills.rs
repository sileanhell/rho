//! Agent Skills discovery and SKILL.md parsing.
//!
//! Frontmatter follows the Agent Skills specification (name, description,
//! license, compatibility, metadata, allowed-tools) and is parsed with a real
//! YAML parser. `disable-model-invocation` is a Rho extension outside the
//! Agent Skills field set; it stays opt-in per skill and is enforced only by
//! prompt metadata injection.
//!
//! Precedence, highest first; the first source for a name wins and conflicts
//! are reported instead of hidden:
//!
//! 1. loose user skills: `~/.rho/skills`, then `~/.agents/skills`
//! 2. loose project skills: nearest `.agents/skills` ancestor first
//! 3. project Agent Plugins skills: nearest `.agents/plugins` ancestor first
//! 4. user Agent Plugins skills: `~/.agents/plugins`

use std::path::{Path, PathBuf};

pub(crate) mod mention;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillSource {
    Filesystem {
        skill_file: PathBuf,
        owner: Option<String>,
    },
}

impl SkillSource {
    fn file(skill_file: PathBuf) -> Self {
        Self::Filesystem {
            skill_file,
            owner: None,
        }
    }

    pub(crate) fn plugin(skill_file: PathBuf, plugin: String) -> Self {
        Self::Filesystem {
            skill_file,
            owner: Some(plugin),
        }
    }
}

impl std::fmt::Display for SkillSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Filesystem {
                skill_file,
                owner: None,
            } => formatter.write_str(&crate::paths::display(skill_file)),
            Self::Filesystem {
                skill_file,
                owner: Some(owner),
            } => write!(
                formatter,
                "plugin {owner} ({})",
                crate::paths::display(skill_file.parent().unwrap_or(skill_file))
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub disable_model_invocation: bool,
    pub source: SkillSource,
    pub contents: String,
}

pub fn discover(cwd: &Path) -> Vec<Skill> {
    let home = crate::paths::home_dir();
    discover_with_home(cwd, home.as_deref())
}

pub fn discover_with_home(cwd: &Path, home: Option<&Path>) -> Vec<Skill> {
    let plugin_skills = crate::plugins::skills_by_precedence(cwd, home);
    discover_with_plugin_skills(cwd, home, plugin_skills)
}

pub(crate) fn discover_with_plugin_skills(
    cwd: &Path,
    home: Option<&Path>,
    plugin_skills: Vec<Skill>,
) -> Vec<Skill> {
    let mut roots = Vec::new();
    if let Some(home) = home {
        roots.extend(crate::paths::user_skill_dirs(home));
    }
    roots.extend(
        crate::workspace::project_ancestor_dirs(cwd)
            .into_iter()
            .rev()
            .map(|path| path.join(".agents").join("skills")),
    );

    // Candidates arrive in precedence order: loose user, loose project,
    // then plugin skills (project plugins before user plugins).
    let mut candidates: Vec<Skill> = roots
        .into_iter()
        .flat_map(|root| skill_paths(&root))
        .filter_map(|path| match read_skill(&path) {
            Ok(skill) => Some(skill),
            Err(error) => {
                tracing::warn!(skill = %path.display(), error = %error, "skipping invalid skill");
                None
            }
        })
        .collect();
    candidates.extend(plugin_skills);

    let mut skills: Vec<Skill> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if let Some(winner) = skills.iter().find(|skill| skill.name == candidate.name) {
            tracing::warn!(
                skill = %candidate.name,
                selected = %winner.source,
                ignored = %candidate.source,
                "duplicate skill name; keeping the higher-precedence source"
            );
            continue;
        }
        skills.push(candidate);
    }
    // Sort after precedence/dedup so every presentation surface shares one order.
    skills.sort_by(|left, right| {
        left.name
            .to_ascii_lowercase()
            .cmp(&right.name.to_ascii_lowercase())
    });
    skills
}

fn skill_paths(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };

    let mut paths: Vec<_> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                Some(path.join("SKILL.md"))
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    paths
}

fn read_skill(path: &Path) -> anyhow::Result<Skill> {
    let contents = std::fs::read_to_string(path)?;
    parse_skill(&contents, SkillSource::file(path.to_path_buf()), Some(path))
}

/// The SKILL.md instructions the model sees: the document after its YAML
/// frontmatter. Name and description already appear in the skill listing,
/// and the remaining fields are client metadata, so loading omits them.
/// Documents without valid frontmatter are returned unchanged.
pub(crate) fn instructions(contents: &str) -> &str {
    crate::frontmatter::split(contents).map_or(contents, |split| {
        split.body.trim_start_matches(['\n', '\r'])
    })
}

/// Parse and validate one SKILL.md document.
///
/// `skill_path` enables the Agent Skills rule that `name` must match the
/// containing directory; built-in skills have no directory and skip it.
pub(crate) fn parse_skill(
    contents: &str,
    source: SkillSource,
    skill_path: Option<&Path>,
) -> anyhow::Result<Skill> {
    let frontmatter: SkillFrontmatter = serde_yaml_ng::from_str(frontmatter_yaml(contents)?)
        .map_err(|error| anyhow::anyhow!("invalid SKILL.md frontmatter: {error}"))?;
    validate_frontmatter(&frontmatter)?;
    if let Some(path) = skill_path {
        let directory_name = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("missing skill directory name"))?;
        if frontmatter.name != directory_name {
            anyhow::bail!("skill name must match directory name");
        }
    }

    Ok(Skill {
        name: frontmatter.name,
        description: frontmatter.description,
        disable_model_invocation: frontmatter
            .disable_model_invocation
            .map(|flag| flag.0)
            .unwrap_or(false),
        source,
        contents: contents.into(),
    })
}

/// The YAML between the frontmatter fences, without the final line
/// terminator so a trailing block scalar (`description: >`) does not gain a
/// newline that would leak into listings and the 1024-character limit.
fn frontmatter_yaml(contents: &str) -> anyhow::Result<&str> {
    use crate::frontmatter::FrontmatterError;
    match crate::frontmatter::split(contents) {
        Ok(split) => Ok(split
            .yaml
            .strip_suffix('\n')
            .map_or(split.yaml, |yaml| yaml.strip_suffix('\r').unwrap_or(yaml))),
        Err(FrontmatterError::MissingOpening) => {
            anyhow::bail!("SKILL.md must start with YAML frontmatter")
        }
        Err(FrontmatterError::MissingClosing) => anyhow::bail!("unterminated YAML frontmatter"),
    }
}

/// Typed Agent Skills frontmatter plus Rho's `disable-model-invocation`
/// extension. Unknown fields are ignored, matching prior behavior and the
/// ecosystem convention that clients may add their own frontmatter fields.
#[derive(serde::Deserialize)]
struct SkillFrontmatter {
    name: String,
    description: String,
    #[expect(dead_code, reason = "type-validated only")]
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    compatibility: Option<String>,
    #[expect(dead_code, reason = "type-validated only")]
    #[serde(rename = "allowed-tools", default)]
    allowed_tools: Option<String>,
    #[serde(rename = "disable-model-invocation", default)]
    disable_model_invocation: Option<TolerantBool>,
}

/// Accepts YAML booleans plus the legacy string forms `"true"`/`"false"`
/// (case-insensitive) that the earlier hand-written parser tolerated.
#[derive(Clone, Copy)]
struct TolerantBool(bool);

impl<'de> serde::Deserialize<'de> for TolerantBool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visitor;

        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = TolerantBool;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a boolean or the string \"true\"/\"false\"")
            }

            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<TolerantBool, E> {
                Ok(TolerantBool(value))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<TolerantBool, E> {
                match value.to_ascii_lowercase().as_str() {
                    "true" => Ok(TolerantBool(true)),
                    "false" => Ok(TolerantBool(false)),
                    _ => Err(serde::de::Error::custom(
                        "disable-model-invocation must be true or false",
                    )),
                }
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}

/// Agent Skills field constraints, separate from YAML syntax parsing.
fn validate_frontmatter(frontmatter: &SkillFrontmatter) -> anyhow::Result<()> {
    validate_name(&frontmatter.name)?;
    let description_chars = frontmatter.description.chars().count();
    if description_chars == 0 || description_chars > 1024 {
        anyhow::bail!("skill description must be 1-1024 characters");
    }
    if let Some(compatibility) = &frontmatter.compatibility {
        let compatibility_chars = compatibility.chars().count();
        if compatibility_chars == 0 || compatibility_chars > 500 {
            anyhow::bail!("skill compatibility must be 1-500 characters");
        }
    }
    // `license` and `allowed-tools` (space-separated tool list) are constrained
    // by their deserialized types. `metadata` and other client fields are ignored.
    Ok(())
}

fn validate_name(name: &str) -> anyhow::Result<()> {
    if name.is_empty() || name.len() > 64 {
        anyhow::bail!("skill name must be 1-64 characters");
    }
    let bytes = name.as_bytes();
    if bytes.first() == Some(&b'-') || bytes.last() == Some(&b'-') || name.contains("--") {
        anyhow::bail!("skill name must use single hyphen separators");
    }
    if !bytes
        .iter()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
    {
        anyhow::bail!("skill name must be lowercase alphanumeric with hyphen separators");
    }
    Ok(())
}
