/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import ts from 'typescript';
import fs from 'fs/promises';
import { existsSync, readdirSync, statSync } from 'fs';
import path from 'path';
import { getWorkspaceDir } from './workspace.js';

export interface ParameterInfo {
  name: string;
  type: string;
  optional: boolean;
  defaultValue?: string;
}

export interface MemberInfo {
  name: string;
  type: string;
  optional?: boolean;
  kind: string;
  line?: number;
}

export interface ObjectDefinition {
  name: string;
  kind: 'function' | 'component' | 'interface' | 'type_alias' | 'class' | 'enum' | 'variable' | 'const';
  isExported: boolean;
  isDefaultExport?: boolean;
  startLine: number;
  endLine: number;
  parameters?: ParameterInfo[];
  returnType?: string;
  typeDefinition?: string;
  extends?: string[];
  members?: MemberInfo[];
  description?: string;
  hooksUsed?: string[];
  subComponentsUsed?: string[];
  stateVariables?: { name: string; type?: string }[];
}

export interface ImportInfo {
  module: string;
  importedSymbols: string[];
  isRelative: boolean;
  line: number;
}

export interface ExportInfo {
  name: string;
  kind?: string;
  isDefault?: boolean;
  line: number;
}

export interface FileKnowledgeNode {
  filePath: string;
  fileName: string;
  extension: string;
  language: string;
  sizeBytes: number;
  lineCount: number;
  imports: ImportInfo[];
  exports: ExportInfo[];
  objectDefinitions: ObjectDefinition[];
}

export interface SourceCodeKnowledgeGraph {
  metadata: {
    title: string;
    description: string;
    generatedAt: string;
    totalFiles: number;
    totalDefinitions: number;
    totalLinesOfCode: number;
    sourceRoot: string;
  };
  summary: {
    filesCount: number;
    componentsCount: number;
    functionsCount: number;
    interfacesCount: number;
    typeAliasesCount: number;
    classesCount: number;
    variablesCount: number;
  };
  files: Record<string, FileKnowledgeNode>;
}

/**
 * Parses a single TypeScript/TSX file and builds its knowledge representation
 */
export function parseSourceFileAST(filePath: string, fileContent: string): FileKnowledgeNode {
  const ext = path.extname(filePath).toLowerCase();
  const fileName = path.basename(filePath);
  const lineCount = fileContent.split('\n').length;
  
  const isTs = ext === '.ts' || ext === '.tsx';
  const isJs = ext === '.js' || ext === '.jsx' || ext === '.mjs' || ext === '.cjs';

  const node: FileKnowledgeNode = {
    filePath,
    fileName,
    extension: ext,
    language: isTs ? (ext === '.tsx' ? 'tsx' : 'typescript') : isJs ? 'javascript' : ext.replace('.', ''),
    sizeBytes: Buffer.byteLength(fileContent, 'utf8'),
    lineCount,
    imports: [],
    exports: [],
    objectDefinitions: []
  };

  if (!isTs && !isJs) {
    // For CSS, JSON, Markdown, etc., create basic file knowledge node
    return node;
  }

  try {
    const scriptKind = ext === '.tsx' || ext === '.jsx' ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
    const sourceFile = ts.createSourceFile(
      filePath,
      fileContent,
      ts.ScriptTarget.Latest,
      true,
      scriptKind
    );

    const getLineNum = (pos: number) => {
      const { line } = sourceFile.getLineAndCharacterOfPosition(pos);
      return line + 1;
    };

    const isNodeExported = (n: ts.Node): boolean => {
      const modifiers = (n as any).modifiers as ts.Modifier[] | undefined;
      if (!modifiers) return false;
      return modifiers.some(m => m.kind === ts.SyntaxKind.ExportKeyword);
    };

    const isNodeDefaultExport = (n: ts.Node): boolean => {
      const modifiers = (n as any).modifiers as ts.Modifier[] | undefined;
      if (!modifiers) return false;
      return modifiers.some(m => m.kind === ts.SyntaxKind.DefaultKeyword);
    };

    const extractJSDoc = (n: ts.Node): string | undefined => {
      const jsDocNodes = (n as any).jsDoc;
      if (jsDocNodes && jsDocNodes.length > 0) {
        return jsDocNodes.map((doc: any) => doc.comment || '').join(' ').trim();
      }
      return undefined;
    };

    const extractParams = (params: ts.NodeArray<ts.ParameterDeclaration>): ParameterInfo[] => {
      return params.map(p => ({
        name: p.name.getText(sourceFile),
        type: p.type ? p.type.getText(sourceFile) : 'any',
        optional: Boolean(p.questionToken || p.initializer),
        defaultValue: p.initializer ? p.initializer.getText(sourceFile) : undefined
      }));
    };

    const extractSubComponentsAndHooks = (bodyNode?: ts.Node) => {
      const hooksUsed = new Set<string>();
      const subComponentsUsed = new Set<string>();
      const stateVariables: { name: string; type?: string }[] = [];

      if (!bodyNode) return { hooksUsed: [], subComponentsUsed: [], stateVariables };

      const visitBody = (child: ts.Node) => {
        // Hooks identification (e.g. useState, useEffect, useWorkspaceFiles)
        if (ts.isCallExpression(child)) {
          const exprText = child.expression.getText(sourceFile);
          if (exprText.startsWith('use') && exprText.length > 3 && exprText[3] === exprText[3].toUpperCase()) {
            hooksUsed.add(exprText);
          }
          if (exprText === 'useState' && child.parent && ts.isVariableDeclaration(child.parent)) {
            const varName = child.parent.name.getText(sourceFile);
            const genericType = child.typeArguments && child.typeArguments[0] ? child.typeArguments[0].getText(sourceFile) : undefined;
            stateVariables.push({ name: varName, type: genericType });
          }
        }
        // JSX Components identification (<WorkspaceHeaderBar />, <ChatPanel />)
        if (ts.isJsxOpeningElement(child) || ts.isJsxSelfClosingElement(child)) {
          const tagName = child.tagName.getText(sourceFile);
          if (tagName[0] === tagName[0].toUpperCase() && tagName !== 'Fragment') {
            subComponentsUsed.add(tagName);
          }
        }
        ts.forEachChild(child, visitBody);
      };

      visitBody(bodyNode);
      return {
        hooksUsed: Array.from(hooksUsed),
        subComponentsUsed: Array.from(subComponentsUsed),
        stateVariables
      };
    };

    // Traverse root AST nodes
    ts.forEachChild(sourceFile, (nodeItem) => {
      // 1. Imports
      if (ts.isImportDeclaration(nodeItem)) {
        const modName = nodeItem.moduleSpecifier.getText(sourceFile).replace(/['"]/g, '');
        const symbols: string[] = [];
        if (nodeItem.importClause) {
          if (nodeItem.importClause.name) {
            symbols.push(`default:${nodeItem.importClause.name.getText(sourceFile)}`);
          }
          if (nodeItem.importClause.namedBindings) {
            if (ts.isNamedImports(nodeItem.importClause.namedBindings)) {
              nodeItem.importClause.namedBindings.elements.forEach(el => {
                symbols.push(el.name.getText(sourceFile));
              });
            } else if (ts.isNamespaceImport(nodeItem.importClause.namedBindings)) {
              symbols.push(`* as ${nodeItem.importClause.namedBindings.name.getText(sourceFile)}`);
            }
          }
        }
        node.imports.push({
          module: modName,
          importedSymbols: symbols,
          isRelative: modName.startsWith('.'),
          line: getLineNum(nodeItem.getStart())
        });
      }

      // 2. Function Declarations
      else if (ts.isFunctionDeclaration(nodeItem)) {
        const fnName = nodeItem.name ? nodeItem.name.getText(sourceFile) : 'anonymous';
        const isExported = isNodeExported(nodeItem);
        const isDefault = isNodeDefaultExport(nodeItem);
        const startLine = getLineNum(nodeItem.getStart());
        const endLine = getLineNum(nodeItem.getEnd());
        const isComponent = fnName[0] === fnName[0].toUpperCase();

        const { hooksUsed, subComponentsUsed, stateVariables } = extractSubComponentsAndHooks(nodeItem.body);

        const def: ObjectDefinition = {
          name: fnName,
          kind: isComponent ? 'component' : 'function',
          isExported,
          isDefaultExport: isDefault,
          startLine,
          endLine,
          parameters: extractParams(nodeItem.parameters),
          returnType: nodeItem.type ? nodeItem.type.getText(sourceFile) : (isComponent ? 'JSX.Element' : 'void'),
          description: extractJSDoc(nodeItem),
          hooksUsed,
          subComponentsUsed,
          stateVariables
        };

        node.objectDefinitions.push(def);
        if (isExported) {
          node.exports.push({ name: fnName, kind: def.kind, isDefault, line: startLine });
        }
      }

      // 3. Interface Declarations
      else if (ts.isInterfaceDeclaration(nodeItem)) {
        const interfaceName = nodeItem.name.getText(sourceFile);
        const isExported = isNodeExported(nodeItem);
        const startLine = getLineNum(nodeItem.getStart());
        const endLine = getLineNum(nodeItem.getEnd());

        const members: MemberInfo[] = nodeItem.members.map(m => {
          const mName = m.name ? m.name.getText(sourceFile) : 'member';
          const mType = (m as any).type ? (m as any).type.getText(sourceFile) : 'any';
          return {
            name: mName,
            type: mType,
            optional: Boolean((m as any).questionToken),
            kind: ts.isPropertySignature(m) ? 'property' : ts.isMethodSignature(m) ? 'method' : 'member',
            line: getLineNum(m.getStart())
          };
        });

        const extendsClauses: string[] = [];
        if (nodeItem.heritageClauses) {
          nodeItem.heritageClauses.forEach(h => {
            h.types.forEach(t => extendsClauses.push(t.getText(sourceFile)));
          });
        }

        const def: ObjectDefinition = {
          name: interfaceName,
          kind: 'interface',
          isExported,
          startLine,
          endLine,
          members,
          extends: extendsClauses.length > 0 ? extendsClauses : undefined,
          description: extractJSDoc(nodeItem)
        };

        node.objectDefinitions.push(def);
        if (isExported) {
          node.exports.push({ name: interfaceName, kind: 'interface', line: startLine });
        }
      }

      // 4. Type Alias Declarations
      else if (ts.isTypeAliasDeclaration(nodeItem)) {
        const typeName = nodeItem.name.getText(sourceFile);
        const isExported = isNodeExported(nodeItem);
        const startLine = getLineNum(nodeItem.getStart());
        const endLine = getLineNum(nodeItem.getEnd());

        const def: ObjectDefinition = {
          name: typeName,
          kind: 'type_alias',
          isExported,
          startLine,
          endLine,
          typeDefinition: nodeItem.type.getText(sourceFile),
          description: extractJSDoc(nodeItem)
        };

        node.objectDefinitions.push(def);
        if (isExported) {
          node.exports.push({ name: typeName, kind: 'type_alias', line: startLine });
        }
      }

      // 5. Enum Declarations
      else if (ts.isEnumDeclaration(nodeItem)) {
        const enumName = nodeItem.name.getText(sourceFile);
        const isExported = isNodeExported(nodeItem);
        const startLine = getLineNum(nodeItem.getStart());
        const endLine = getLineNum(nodeItem.getEnd());

        const members: MemberInfo[] = nodeItem.members.map(m => ({
          name: m.name.getText(sourceFile),
          type: m.initializer ? m.initializer.getText(sourceFile) : 'auto',
          kind: 'enum_member',
          line: getLineNum(m.getStart())
        }));

        const def: ObjectDefinition = {
          name: enumName,
          kind: 'enum',
          isExported,
          startLine,
          endLine,
          members,
          description: extractJSDoc(nodeItem)
        };

        node.objectDefinitions.push(def);
        if (isExported) {
          node.exports.push({ name: enumName, kind: 'enum', line: startLine });
        }
      }

      // 6. Class Declarations
      else if (ts.isClassDeclaration(nodeItem)) {
        const className = nodeItem.name ? nodeItem.name.getText(sourceFile) : 'AnonymousClass';
        const isExported = isNodeExported(nodeItem);
        const isDefault = isNodeDefaultExport(nodeItem);
        const startLine = getLineNum(nodeItem.getStart());
        const endLine = getLineNum(nodeItem.getEnd());

        const members: MemberInfo[] = nodeItem.members.map(m => ({
          name: m.name ? m.name.getText(sourceFile) : 'constructor',
          type: (m as any).type ? (m as any).type.getText(sourceFile) : 'any',
          kind: ts.isMethodDeclaration(m) ? 'method' : ts.isPropertyDeclaration(m) ? 'property' : 'constructor',
          line: getLineNum(m.getStart())
        }));

        const def: ObjectDefinition = {
          name: className,
          kind: 'class',
          isExported,
          isDefaultExport: isDefault,
          startLine,
          endLine,
          members,
          description: extractJSDoc(nodeItem)
        };

        node.objectDefinitions.push(def);
        if (isExported) {
          node.exports.push({ name: className, kind: 'class', isDefault, line: startLine });
        }
      }

      // 7. Variable Statements (const/let/var, arrow functions, React components)
      else if (ts.isVariableStatement(nodeItem)) {
        const isExported = isNodeExported(nodeItem);
        const startLine = getLineNum(nodeItem.getStart());
        const endLine = getLineNum(nodeItem.getEnd());

        nodeItem.declarationList.declarations.forEach(decl => {
          const varName = decl.name.getText(sourceFile);
          let kind: ObjectDefinition['kind'] = 'const';
          let parameters: ParameterInfo[] | undefined;
          let returnType: string | undefined;
          let hooksUsed: string[] | undefined;
          let subComponentsUsed: string[] | undefined;
          let stateVariables: { name: string; type?: string }[] | undefined;

          if (decl.initializer && (ts.isArrowFunction(decl.initializer) || ts.isFunctionExpression(decl.initializer))) {
            const isComp = varName[0] === varName[0].toUpperCase();
            kind = isComp ? 'component' : 'function';
            parameters = extractParams(decl.initializer.parameters);
            returnType = decl.initializer.type ? decl.initializer.type.getText(sourceFile) : (isComp ? 'JSX.Element' : 'void');

            const extracted = extractSubComponentsAndHooks(decl.initializer.body);
            hooksUsed = extracted.hooksUsed;
            subComponentsUsed = extracted.subComponentsUsed;
            stateVariables = extracted.stateVariables;
          }

          const def: ObjectDefinition = {
            name: varName,
            kind,
            isExported,
            startLine,
            endLine,
            parameters,
            returnType: decl.type ? decl.type.getText(sourceFile) : returnType,
            description: extractJSDoc(nodeItem),
            hooksUsed,
            subComponentsUsed,
            stateVariables
          };

          node.objectDefinitions.push(def);
          if (isExported) {
            node.exports.push({ name: varName, kind, line: startLine });
          }
        });
      }

      // 8. Explicit Export Declarations (e.g. export { a, b })
      else if (ts.isExportDeclaration(nodeItem)) {
        if (nodeItem.exportClause && ts.isNamedExports(nodeItem.exportClause)) {
          nodeItem.exportClause.elements.forEach(el => {
            node.exports.push({
              name: el.name.getText(sourceFile),
              line: getLineNum(nodeItem.getStart())
            });
          });
        }
      }

      // 9. Default export assignments (e.g. export default App;)
      else if (ts.isExportAssignment(nodeItem)) {
        node.exports.push({
          name: nodeItem.expression.getText(sourceFile),
          isDefault: true,
          line: getLineNum(nodeItem.getStart())
        });
      }
    });

  } catch (err) {
    console.warn(`AST Parse error for ${filePath}:`, err);
  }

  return node;
}

/**
 * Scans workspace directories recursively and builds full application Knowledge Graph
 */
export async function generateKnowledgeGraph(options?: { saveToFile?: boolean }): Promise<SourceCodeKnowledgeGraph> {
  const rootDir = process.cwd();
  const workspaceDir = getWorkspaceDir();

  const targetDirs = [
    path.join(rootDir, 'src'),
    path.join(rootDir, 'server.ts')
  ];

  const filesMap: Record<string, FileKnowledgeNode> = {};
  let totalDefinitions = 0;
  let totalLinesOfCode = 0;

  const counts = {
    filesCount: 0,
    componentsCount: 0,
    functionsCount: 0,
    interfacesCount: 0,
    typeAliasesCount: 0,
    classesCount: 0,
    variablesCount: 0
  };

  async function collectFiles(dirOrFile: string) {
    if (!existsSync(dirOrFile)) return;
    const stat = statSync(dirOrFile);

    if (stat.isFile()) {
      await processFile(dirOrFile);
      return;
    }

    const items = readdirSync(dirOrFile);
    for (const item of items) {
      if (item.startsWith('.') || item === 'node_modules' || item === 'dist' || item === 'build') continue;
      const full = path.join(dirOrFile, item);
      const childStat = statSync(full);
      if (childStat.isDirectory()) {
        await collectFiles(full);
      } else if (childStat.isFile()) {
        await processFile(full);
      }
    }
  }

  async function processFile(filePath: string) {
    const relPath = path.relative(rootDir, filePath).replace(/\\/g, '/');
    if (relPath.endsWith('eka_src_code_knowledge_graph.json')) return;

    try {
      const content = await fs.readFile(filePath, 'utf8');
      const node = parseSourceFileAST(relPath, content);
      filesMap[relPath] = node;

      counts.filesCount++;
      totalLinesOfCode += node.lineCount;

      node.objectDefinitions.forEach(def => {
        totalDefinitions++;
        if (def.kind === 'component') counts.componentsCount++;
        else if (def.kind === 'function') counts.functionsCount++;
        else if (def.kind === 'interface') counts.interfacesCount++;
        else if (def.kind === 'type_alias') counts.typeAliasesCount++;
        else if (def.kind === 'class') counts.classesCount++;
        else counts.variablesCount++;
      });
    } catch (err) {
      console.error(`Error reading ${relPath} for Knowledge Graph:`, err);
    }
  }

  for (const t of targetDirs) {
    await collectFiles(t);
  }

  const knowledgeGraph: SourceCodeKnowledgeGraph = {
    metadata: {
      title: 'Eka Source Code Knowledge Graph',
      description: 'Comprehensive AST Knowledge Graph containing file names, object definitions, parameters, line numbers, imports, exports, and relationships for entire Eka application.',
      generatedAt: new Date().toISOString(),
      totalFiles: counts.filesCount,
      totalDefinitions,
      totalLinesOfCode,
      sourceRoot: 'src'
    },
    summary: counts,
    files: filesMap
  };

  if (options?.saveToFile !== false) {
    const jsonStr = JSON.stringify(knowledgeGraph, null, 2);
    const targetFileRoot = path.join(rootDir, 'eka_src_code_knowledge_graph.json');
    await fs.writeFile(targetFileRoot, jsonStr, 'utf8');

    if (workspaceDir && workspaceDir !== rootDir) {
      const targetFileWs = path.join(workspaceDir, 'eka_src_code_knowledge_graph.json');
      await fs.writeFile(targetFileWs, jsonStr, 'utf8').catch(() => {});
    }

    console.log(`[Knowledge Graph] Successfully updated eka_src_code_knowledge_graph.json (${counts.filesCount} files, ${totalDefinitions} object definitions)`);
  }

  return knowledgeGraph;
}
